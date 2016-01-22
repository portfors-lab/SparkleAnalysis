@echo off
setlocal EnableExtensions EnableDelayedExpansion
cls

:: ------------------------------------------------
::
:: Place your path to Sparkle Analysis here
set location="C:\Users\Name\Documents\SparkleAnalysis\"
cd %location%
::
:: Place your path to Git here
set gitLocation="C:\Program Files (x86)\Git\bin"
SET PATH=%PATH%;%gitLocation%
::
:: ------------------------------------------------

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