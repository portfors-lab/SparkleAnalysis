@echo off
setlocal EnableExtensions EnableDelayedExpansion
cls

:: --------------------------------------------------------
::
:: HOW TO USE:
:: To use this .bat file for running and updating Sparkle
:: Analysis, make a copy of this file and rename it
:: something easy to remember (e.g. SparkleAnalysis.bat).
:: Now in the copy of this file you will want to change the
:: variable of "location" (line 24) with the path to the
:: SparkleAnalysis directory (where you found this file).
:: After you complete that you will want to change the
:: variable for "gitLocation" (line 28) with the path to
:: where your git.exe is stored. Once you have those set,
:: you can either create a shortcut to your newly edited
:: file or just run the .bat file. While running it with
:: this code, it will check for updates before running
:: and will ask the user if they wish to update if there
:: are any new updates.
::
:: Place your path to Sparkle Analysis here
set location="C:\Users\Name\Documents\SparkleAnalysis\"
cd %location%
::
:: Place your path to Git here
set gitLocation="C:\Program Files (x86)\Git\bin"
SET PATH=%PATH%;%gitLocation%
::
:: --------------------------------------------------------

title Sparkle Analysis

:: Get versions of Sparkle Analysis
git remote update
for /f "delims=" %%i in ('git rev-parse @{0}') do set local=%%i
for /f "delims=" %%i in ('git rev-parse origin/master') do set remote=%%i
for /f "delims=" %%i in ('git merge-base @ origin/master') do set base=%%i

echo.

:: Check relation of various versions
if "%local%" equ "%remote%" (
    echo Sparkle Analysis is up to date.
    goto :runSparkleAnalysis
) else if "%local%" equ "%base%" (
    echo A newer version of Sparkle Analysis is avaliable.
) else if "%remote%" equ "%base%" (
    echo Your local branch of SparkleAnalysis is ahead of origin/master.
    goto :runSparkleAnalysis
)

set "answer=%globalparam1%"
goto :answerCheck

:updatePrompt
set /p "answer=Update Sparkle Analysis? (y or n): "
goto :answerCheck

:answerCheck
if not defined answer goto :updatePrompt

echo.

if "%answer%" == "y" (
    git pull
)

:runSparkleAnalysis
echo.
python run.py