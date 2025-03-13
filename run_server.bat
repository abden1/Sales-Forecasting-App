@echo off 
title Sales Forecasting Server 
cd /d "C:\SAAS\depi\" 
call venv\Scripts\activate.bat 
set PYTHONPATH=C:\SAAS\depi;%PYTHONPATH% 
python wsgi.py 
pause 
