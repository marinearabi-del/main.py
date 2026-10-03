"""
ORG PRO - Oriental Organ
Real-time DSP engine: polyphonic voices, ADSR, real pitch-bend, vibrato/mod wheel,
reverb, sample-accurate rhythm, recording.

Desktop : pip install kivy numpy sounddevice
Android : buildozer requirements = python3,kivy,numpy,pyjnius
"""
import os
import time
import wave
import threading
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.utils import platform
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.relativelayout import RelativeLayout

SR = 44100
BLOCK = 1024 if platform == 'android' else 512
TWO_PI = 2 * np.pi
MAX_VOICES = 8 if platform == 'android' else 16
HCAP = 6 if platform == 'android' else 14   # أقصى عدد هارمونيكات (توفير للمعالج على الهاتف)

NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
WHITE = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23,
         'G': 392.00, 'A': 440.00, 'B': 493.88}
BLACK = {'C#': 277.18, 'D#': 311.13, 'F#': 369.99, 'G#': 415.30, 'A#': 466.16}
BLACK_POS = {'C#': 1, 'D#': 2, 'F#': 4, 'G#': 5, 'A#': 6}

# النغمات التي تُخفض ربع تون لكل مقام (أسماء لاتينية لأن خط Kivy الافتراضي لا يعرض العربية)
MAQAMAT = {
    'Ajam': [],
    'Rast': ['E', 'B'],
    'Bayati': ['E'],
    'Sika': ['E', 'B'],
    'Saba': ['E', 'G'],
}

# D = دم ، T = تك ، S = تك خفيف ، . = سكتة
RHYTHMS = {
    'Maqsum': 'DT.TD.T.',
    'Baladi': 'DD.TD.T.',
    'Saidi': 'DT.DD.T.',
    'Wahda': 'D..TD.T.',
    'Ayyub': 'D..DT.T.',
}


# ======================================================================
#  تعريف الأصوات
# ======================================================================
def harm(fn, n=14):
    return [fn(h) for h in range(1, n + 1)]


INSTR = {
    'Mizmar': dict(amps=harm(lambda h: 1 / h ** 0.7), detune=[0], vib=12,
                   adsr=(0.02, 0.05, 0.85, 0.08), d0=0, dh=0,
                   noise=0.012, nd=0, pluck=False),
    'Oud': dict(amps=harm(lambda h: 1 / h), detune=[0], vib=0,
                adsr=(0.003, 0.001, 1.0, 0.6), d0=1.4, dh=1.3,
                noise=0.25, nd=90, pluck=True),
    'Nay': dict(amps=[1, .25, .08], detune=[0], vib=18,
                adsr=(0.07, 0.05, 0.9, 0.12), d0=0, dh=0,
                noise=0.05, nd=0, pluck=False),
    'Organ': dict(amps=[1, .8, .5, .6, 0, .3, 0, .2], detune=[0], vib=0,
                  adsr=(0.01, 0.0, 1.0, 0.05), d0=0, dh=0,
                  noise=0, nd=0, pluck=False),
    'Strings': dict(amps=harm(lambda h: 1 / h, 8), detune=[-7, 0, 7], vib=6,
                    adsr=(0.15, 0.1, 0.9, 0.3), d0=0, dh=0,
                    noise=0, nd=0, pluck=False),
}


# ======================================================================
#  المؤثرات
# ======================================================================
class Comb:
    def __init__(self, d, g):
        self.d, self.g = d, g
        self.buf = np.zeros(d)
        self.pos = 0

    def process(self, x):
        n = len(x)
        out = np.empty(n)
        i = 0
        while i < n:
            m = min(self.d - self.pos, n - i)
            seg = x[i:i + m] + self.g * self.buf[self.pos:self.pos + m]
            self.buf[self.pos:self.pos + m] = seg
            out[i:i + m] = seg
            self.pos = (self.pos + m) % self.d
            i += m
        return out


class Allpass:
    def __init__(self, d, g):
        self.d, self.g = d, g
        self.buf = np.zeros(d)
        self.pos = 0

    def process(self, x):
        n = len(x)
        out = np.empty(n)
        i = 0
        while i < n:
            m = min(self.d - self.pos, n - i)
            vd = self.buf[self.pos:self.pos + m].copy()
            v = x[i:i + m] + self.g * vd
            self.buf[self.pos:self.pos + m] = v
            out[i:i + m] = -self.g * v + vd
            self.pos = (self.pos + m) % self.d
            i += m
        return out


class Reverb:
    def __init__(self):
        self.combs = [Comb(d, 0.82) for d in (1116, 1188, 1277, 1356)]
        self.aps = [Allpass(556, 0.5), Allpass(441, 0.5)]

    def process(self, x):
        y = sum(c.process(x) for c in self.combs) * 0.25
        for a in self.aps:
            y = a.process(y)
        return y


# ======================================================================
#  الطبول
# ======================================================================
def make_drums():
    rng = np.random.default_rng(1)
    # دم
    n = int(SR * 0.32)
    t = np.arange(n) / SR
    ph = TWO_PI * np.cumsum(50 + 95 * np.exp(-t * 18)) / SR
    dum = np.sin(ph) * np.exp(-t * 9)
    # تك
    n = int(SR * 0.14)
    t = np.arange(n) / SR
    tak = (rng.standard_normal(n) * 0.6 + np.sin(TWO_PI * 1900 * t)) * np.exp(-t * 55)
    out = {}
    for k, arr, g in (('D', dum, 1.0), ('T', tak, 0.75), ('S', tak, 0.35)):
        out[k] = (arr / np.max(np.abs(arr)) * g).astype(np.float64)
    return out


# ======================================================================
#  النغمة (Voice)
# ======================================================================
class Voice:
    def __init__(self, key, freq, p):
        self.key, self.f, self.p = key, freq, p
        self.phase = np.zeros(len(p['detune']))
        self.age = 0
        self.released = False
        self.rel_age = 0
        self.rel_level = 0.0
        self.level = 0.0
        self.rel_s = max(1, int(p['adsr'][3] * SR))
        self.dead = False

    def release(self, fast=False):
        if self.released:
            return
        self.released = True
        self.rel_level = self.level
        self.rel_age = 0
        if fast:
            self.rel_s = int(0.02 * SR)

    def render(self, n, bend, mod):
        p = self.p
        a, d, s, _ = p['adsr']
        a_s, d_s = max(1, int(a * SR)), max(1, int(d * SR))
        idx = np.arange(n)
        ages = self.age + idx
        t = ages / SR

        if not self.released:
            env = np.where(ages < a_s, ages / a_s,
                           np.where(ages < a_s + d_s,
                                    1 - (1 - s) * (ages - a_s) / d_s, s))
            self.level = float(env[-1])
        else:
            env = self.rel_level * np.clip(1 - (self.rel_age + idx) / self.rel_s, 0, 1)
            self.rel_age += n
            if self.rel_age >= self.rel_s:
                self.dead = True

        depth = p['vib'] * np.minimum(t / 0.4, 1.0) + mod * 40.0
        lfo = np.sin(TWO_PI * 5.5 * t)
        mult = bend * 2 ** (depth * lfo / 1200.0)

        amps = p['amps']
        hmax = max(1, min(len(amps), HCAP, int((SR / 2 - 2000) // (self.f * 1.25))))
        sig = np.zeros(n)
        decaying = p['d0'] > 0 or p['dh'] > 0
        for k, det in enumerate(p['detune']):
            fk = self.f * 2 ** (det / 1200.0) * mult
            ph = self.phase[k] + np.cumsum(TWO_PI * fk / SR)
            self.phase[k] = ph[-1] % TWO_PI
            for h in range(1, hmax + 1):
                amp = amps[h - 1]
                if amp <= 0.001:
                    continue
                if decaying:
                    sig += amp * np.sin(h * ph) * np.exp(-t * (p['d0'] + p['dh'] * h))
                else:
                    sig += amp * np.sin(h * ph)
        if len(p['detune']) > 1:
            sig /= len(p['detune'])
        if p['noise']:
            nz = np.random.standard_normal(n) * p['noise']
            if p['nd']:
                nz *= np.exp(-t * p['nd'])
            sig += nz

        self.age += n
        if p['pluck'] and self.age > SR * 5:
            self.dead = True
        return sig * env * 0.22


# ======================================================================
#  المحرك
# ======================================================================
class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.voices = []
        self.inst = 'Mizmar'
        self.volume = 0.8
        self.rev_mix = 0.25
        self.drum_vol = 0.7
        self.bend_cents = 0.0
        self._bend_prev = 0.0
        self.mod = 0.0
        self.sustain = False
        self.pedal_held = set()
        self.peak = 0.0
        self.rec = None
        self.error = None
        self.reverb = Reverb()
        self.drums = make_drums()
        self.active_drums = []
        self.rhythm_on = False
        self.pattern = RHYTHMS['Maqsum']
        self.bpm = 100
        self.countdown = 0.0
        self.step = 0

    # ---- واجهة العزف ----
    def note_on(self, key, freq):
        with self.lock:
            self.pedal_held.discard(key)
            for v in self.voices:
                if v.key is key:
                    v.release()
            if len(self.voices) >= MAX_VOICES:
                self.voices[0].release(fast=True)
            self.voices.append(Voice(key, freq, INSTR[self.inst]))

    def note_off(self, key):
        with self.lock:
            if self.sustain:
                self.pedal_held.add(key)
            else:
                self._release_key(key)

    def _release_key(self, key):
        for v in self.voices:
            if v.key is key:
                v.release()

    def set_sustain(self, on):
        with self.lock:
            self.sustain = on
            if not on:
                for k in self.pedal_held:
                    self._release_key(k)
                self.pedal_held.clear()

    def panic(self):
        with self.lock:
            self.voices.clear()
            self.active_drums.clear()
            self.pedal_held.clear()

    # ---- الإيقاع ----
    def set_rhythm(self, on):
        with self.lock:
            self.rhythm_on = on
            self.countdown = 0.0
            self.step = 0

    # ---- التسجيل ----
    def start_record(self):
        with self.lock:
            self.rec = []

    def stop_record(self, path):
        with self.lock:
            chunks, self.rec = self.rec, None
        if not chunks:
            return None
        data = np.concatenate(chunks)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(data.tobytes())
        return path

    # ---- توليد الصوت (يُستدعى من خيط الصوت) ----
    def render(self, n):
        try:
            return self._render(n)
        except Exception as e:  # لا نسمح لخطأ بإيقاف التيار الصوتي
            self.error = repr(e)
            return np.zeros(n, dtype=np.float32)

    def _render(self, n):
        with self.lock:
            cur = self._bend_prev + (self.bend_cents - self._bend_prev) * 0.5
            bend = 2 ** (np.linspace(self._bend_prev, cur, n) / 1200.0)
            self._bend_prev = cur

            out = np.zeros(n)
            for v in self.voices:
                out += v.render(n, bend, self.mod)
            self.voices = [v for v in self.voices if not v.dead]

            drum_out = np.zeros(n)
            if self.rhythm_on:
                step_len = SR * 60.0 / self.bpm / 2
                while self.countdown < n:
                    c = self.pattern[self.step % len(self.pattern)]
                    self.step += 1
                    if c in self.drums:
                        self.active_drums.append([self.drums[c], -int(self.countdown)])
                    self.countdown += step_len
                self.countdown -= n
            alive = []
            for dv in self.active_drums:
                arr, pos = dv
                i0, s0 = max(0, -pos), max(0, pos)
                take = min(n - i0, len(arr) - s0)
                if take > 0:
                    drum_out[i0:i0 + take] += arr[s0:s0 + take]
                dv[1] = pos + n
                if dv[1] < len(arr):
                    alive.append(dv)
            self.active_drums = alive

            dry = out * self.volume
            mix = dry
            if self.rev_mix > 0.01:
                mix = dry + self.reverb.process(dry) * self.rev_mix
            mix = mix + drum_out * self.drum_vol * self.volume * 0.8
            mix = np.tanh(mix)  # ليمتر ناعم يمنع التشويه

            pk = float(np.max(np.abs(mix)))
            self.peak = max(pk, self.peak * 0.9)
            if self.rec is not None:
                self.rec.append((mix * 32767).astype(np.int16))
            return mix.astype(np.float32)


# ======================================================================
#  مخارج الصوت
# ======================================================================
class DesktopBackend:
    def __init__(self, engine):
        self.engine = engine
        self.stream = None

    def start(self):
        import sounddevice as sd

        def cb(outdata, frames, tinfo, status):
            outdata[:, 0] = self.engine.render(frames)

        self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype='float32',
                                      blocksize=BLOCK, callback=cb)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class AndroidBackend:
    def __init__(self, engine):
        self.engine = engine
        self.running = False
        self.track = None

    def start(self):
        from jnius import autoclass
        AudioTrack = autoclass('android.media.AudioTrack')
        AudioManager = autoclass('android.media.AudioManager')
        AudioFormat = autoclass('android.media.AudioFormat')
        ch = AudioFormat.CHANNEL_OUT_MONO
        enc = AudioFormat.ENCODING_PCM_16BIT
        minbuf = AudioTrack.getMinBufferSize(SR, ch, enc)
        self.track = AudioTrack(AudioManager.STREAM_MUSIC, SR, ch, enc,
                                max(minbuf, BLOCK * 4), AudioTrack.MODE_STREAM)
        self.track.play()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        try:
            while self.running:
                buf = self.engine.render(BLOCK)
                pcm = (np.clip(buf, -1, 1) * 32767).astype('<i2')
                self.track.write(pcm.view(np.int8).tolist(), 0, len(pcm) * 2)
        except Exception as e:
            self.engine.error = "Audio thread: " + repr(e)

    def stop(self):
        self.running = False
        if self.track:
            self.track.stop()
            self.track.release()


# ======================================================================
#  الواجهة
# ======================================================================
class SpringSlider(Slider):
    """عجلة البيتش: ترجع للمنتصف عند رفع الإصبع"""

    def on_touch_up(self, touch):
        r = super().on_touch_up(touch)
        if touch.grab_current is self:
            self.value = 0
        return r


class OrgPro(App):
    def build(self):
        self.title = "ORG PRO"
        self.engine = Engine()
        self.octave = 0
        self.transpose = 0
        self.maqam = 'Ajam'
        self.rhythm_name = 'Maqsum'
        self.quarter = {k: False for k in NOTE_NAMES}
        self.q_buttons = {}
        self.rec_path = ""
        self.status = "starting..."

        root = BoxLayout(orientation='vertical', padding=5, spacing=4)

        # شاشة + مقياس المستوى
        top = BoxLayout(orientation='vertical', size_hint_y=0.14)
        self.lcd = Label(text="", font_size='13sp', halign='center',
                         color=(0.1, 1, 0.1, 1))
        self.meter = ProgressBar(max=1.0, value=0, size_hint_y=0.15)
        top.add_widget(self.lcd)
        top.add_widget(self.meter)
        root.add_widget(top)

        # صف 1: الصوت + المقام
        r1 = BoxLayout(size_hint_y=0.08, spacing=4)
        sp = Spinner(text='Mizmar', values=list(INSTR), font_size='13sp')
        sp.bind(text=lambda i, v: self.set_inst(v))
        r1.add_widget(sp)
        sp2 = Spinner(text='Ajam', values=list(MAQAMAT), font_size='13sp')
        sp2.bind(text=self.on_maqam)
        r1.add_widget(sp2)
        root.add_widget(r1)

        # صف 2: أوكتاف / ترانسبوز / سستين
        r2 = BoxLayout(size_hint_y=0.08, spacing=4)
        for txt, fn in (("OCT -", lambda: self.shift_oct(-1)),
                        ("OCT +", lambda: self.shift_oct(1)),
                        ("TRP -", lambda: self.shift_trp(-1)),
                        ("TRP +", lambda: self.shift_trp(1))):
            b = Button(text=txt, font_size='12sp')
            b.bind(on_press=lambda i, f=fn: f())
            r2.add_widget(b)
        sus = ToggleButton(text="SUSTAIN", font_size='12sp')
        sus.bind(state=lambda i, s: self.engine.set_sustain(s == 'down'))
        r2.add_widget(sus)
        root.add_widget(r2)

        # صف 3: سلايدرات
        r3 = BoxLayout(size_hint_y=0.08, spacing=4)
        self._add_slider(r3, "VOL", 0, 1, self.engine.volume, lambda v: setattr(self.engine, 'volume', v))
        self._add_slider(r3, "REV", 0, 1, self.engine.rev_mix, lambda v: setattr(self.engine, 'rev_mix', v))
        self._add_slider(r3, "DRUM", 0, 1, self.engine.drum_vol, lambda v: setattr(self.engine, 'drum_vol', v))
        root.add_widget(r3)

        # صف 4: ربع التون
        r4 = GridLayout(rows=1, spacing=2, size_hint_y=0.10)
        for n in NOTE_NAMES:
            b = ToggleButton(text=n + "\n1/4", font_size='11sp',
                             background_color=(0.18, 0.18, 0.18, 1))
            b.bind(state=lambda inst, st, n=n: self.set_quarter(inst, st, n))
            self.q_buttons[n] = b
            r4.add_widget(b)
        root.add_widget(r4)

        # صف 5: إيقاع + تسجيل
        r5 = BoxLayout(size_hint_y=0.08, spacing=4)
        rb = ToggleButton(text="RHYTHM", font_size='12sp')
        rb.bind(state=lambda i, s: self.engine.set_rhythm(s == 'down'))
        r5.add_widget(rb)
        sp3 = Spinner(text='Maqsum', values=list(RHYTHMS), font_size='13sp')
        sp3.bind(text=self.on_rhythm)
        r5.add_widget(sp3)
        self._add_slider(r5, "BPM", 60, 180, 100, self.on_bpm, step=2)
        rec = ToggleButton(text="REC", font_size='12sp',
                           background_color=(0.8, 0.2, 0.2, 1))
        rec.bind(state=self.on_rec)
        r5.add_widget(rec)
        root.add_widget(r5)

        # الكيبورد + عجلات
        kb_row = BoxLayout(size_hint_y=0.36, spacing=3)
        pb = BoxLayout(orientation='vertical', size_hint_x=0.07)
        pb.add_widget(Label(text="PB", font_size='9sp', size_hint_y=0.1))
        pw = SpringSlider(min=-1, max=1, value=0, orientation='vertical')
        pw.bind(value=lambda i, v: setattr(self.engine, 'bend_cents', v * 200.0))
        pb.add_widget(pw)
        kb_row.add_widget(pb)
        mb = BoxLayout(orientation='vertical', size_hint_x=0.07)
        mb.add_widget(Label(text="MOD", font_size='9sp', size_hint_y=0.1))
        mw = Slider(min=0, max=1, value=0, orientation='vertical')
        mw.bind(value=lambda i, v: setattr(self.engine, 'mod', v))
        mb.add_widget(mw)
        kb_row.add_widget(mb)

        kb = RelativeLayout()
        white = GridLayout(rows=1, spacing=2, size_hint=(1, 1),
                           pos_hint={'x': 0, 'y': 0})
        for o in range(2):
            for n in NOTE_NAMES:
                b = Button(text=n if o == 0 else n + "'", font_size='14sp',
                           background_color=(1, 1, 1, 1), color=(0, 0, 0, 1),
                           always_release=True)
                b.bind(on_press=lambda inst, n=n, o=o: self.key_on(inst, WHITE[n] * 2 ** o, n))
                b.bind(on_release=lambda inst: self.engine.note_off(inst))
                white.add_widget(b)
        kb.add_widget(white)
        kw, bw = 1 / 14, 0.04
        for o in range(2):
            for name, idx in BLACK_POS.items():
                b = Button(text=name, font_size='9sp',
                           background_color=(0, 0, 0, 1), color=(1, 1, 1, 1),
                           size_hint=(bw, 0.58),
                           pos_hint={'x': (idx + 7 * o) * kw - bw / 2, 'y': 0.42},
                           always_release=True)
                b.bind(on_press=lambda inst, f=BLACK[name] * 2 ** o: self.key_on(inst, f, None))
                b.bind(on_release=lambda inst: self.engine.note_off(inst))
                kb.add_widget(b)
        kb_row.add_widget(kb)
        root.add_widget(kb_row)

        Clock.schedule_once(self.start_audio, 0.2)
        Clock.schedule_interval(self.update_ui, 0.08)
        return root

    # ---------- مساعدات واجهة ----------
    def _add_slider(self, parent, label, lo, hi, val, cb, step=0):
        parent.add_widget(Label(text=label, size_hint_x=0.14, font_size='10sp'))
        s = Slider(min=lo, max=hi, value=val, step=step)
        s.bind(value=lambda i, v: cb(v))
        parent.add_widget(s)

    def start_audio(self, dt):
        try:
            self.backend = AndroidBackend(self.engine) if platform == 'android' \
                else DesktopBackend(self.engine)
            self.backend.start()
            self.status = "AUDIO OK"
        except ImportError as e:
            self.status = "Missing: " + str(e) + (" (pip install sounddevice)" if platform != 'android' else "")
        except Exception as e:
            self.status = "Audio error: " + repr(e)

    def on_stop(self):
        try:
            self.backend.stop()
        except Exception:
            pass

    def update_ui(self, dt):
        e = self.engine
        self.meter.value = min(1.0, e.peak)
        rec = "  ● REC" if e.rec is not None else ""
        err = ("\nERR: " + e.error) if e.error else ""
        self.lcd.text = ("ORG PRO  |  %s  |  Maqam: %s%s\nOct %+d   Transpose %+d   Rhythm %s %d BPM   Voices %d\n%s %s%s"
                         % (e.inst, self.maqam, rec, self.octave, self.transpose,
                            self.rhythm_name, e.bpm, len(e.voices),
                            self.status, self.rec_path, err))

    # ---------- التحكم ----------
    def set_inst(self, name):
        self.engine.inst = name

    def shift_oct(self, d):
        self.octave = max(-2, min(2, self.octave + d))

    def shift_trp(self, d):
        self.transpose = max(-12, min(12, self.transpose + d))

    def on_maqam(self, sp, name):
        self.maqam = name
        for n, b in self.q_buttons.items():
            b.state = 'down' if n in MAQAMAT[name] else 'normal'

    def set_quarter(self, inst, state, note):
        on = state == 'down'
        self.quarter[note] = on
        inst.background_color = (1, 0.5, 0, 1) if on else (0.18, 0.18, 0.18, 1)

    def on_rhythm(self, sp, name):
        self.rhythm_name = name
        with self.engine.lock:
            self.engine.pattern = RHYTHMS[name]
            self.engine.step = 0

    def on_bpm(self, v):
        self.engine.bpm = int(v)

    def on_rec(self, btn, state):
        if state == 'down':
            self.engine.start_record()
            self.rec_path = ""
        else:
            path = os.path.join(self.user_data_dir,
                                time.strftime("rec_%Y%m%d_%H%M%S.wav"))
            saved = self.engine.stop_record(path)
            self.rec_path = ("Saved: " + saved) if saved else ""

    def key_on(self, btn, freq, note):
        if note is not None and self.quarter.get(note):
            freq *= 2 ** (-50 / 1200)
        freq *= 2 ** (self.octave + self.transpose / 12.0)
        self.engine.note_on(btn, freq)


if __name__ == '__main__':
    OrgPro().run()
