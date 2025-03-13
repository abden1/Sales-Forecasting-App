Set WshShell = CreateObject("WScript.Shell") 
WshShell.Run "taskkill /f /im pythonw.exe", 0, True 
WshShell.Run "wscript.exe silent_run.vbs", 0, True 
WScript.Quit 
