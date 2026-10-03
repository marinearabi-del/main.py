[app]
title = ORG PRO
package.name = orgpro
package.domain = org.orgpro
source.dir = .
source.include_exts = py
version = 0.1
requirements = python3,kivy,numpy,pyjnius
orientation = portrait
fullscreen = 0

android.archs = arm64-v8a
android.api = 33
android.minapi = 21
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
