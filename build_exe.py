import PyInstaller.__main__
import sys
import os

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

PyInstaller.__main__.run([
    'app/app.py',
    '--onefile',
    '--clean',
    '--add-data=app/templates;templates',
    '--add-data=app/static;static',
    '--name=SalesForecasting',
    '--icon=app/static/favicon.ico',
    '--noconsole',
]) 