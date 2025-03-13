Set WshShell = CreateObject("WScript.Shell") 
WshShell.CurrentDirectory = "C:\SAAS\depi" 
WshShell.Run "pythonw wsgi.py", 0, False 
WScript.Quit 
