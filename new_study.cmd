@echo off
title NEW OpTuna TimeSeries Forecats study
REM Delete the folder "lightning_logs" and its contents
IF EXIST "lightning_logs" (
    rmdir /s /q "lightning_logs"
)

REM Delete all folders that start with "chronos_model_"
for /d %%F in ("chronos_model_*") do (
    rmdir /s /q "%%F"
)

REM Delete the file "my_study.db"
IF EXIST "my_study.db" (
    del /q "my_study.db"
)

REM Call the Python script
python "tune_opt_forecats.py"

REM Pause to keep window open if run by double-click
pause