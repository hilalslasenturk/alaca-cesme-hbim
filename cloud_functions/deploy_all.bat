@echo off
REM Deploy all Scan-to-HBIM V6 Cloud Functions
REM Usage: deploy_all.bat

set PROJECT_ID=concrete-racer-470219-h8
set REGION=europe-west1
set RUNTIME=python311
set MEMORY=2048MB
set TIMEOUT=540s

echo ==============================================
echo Deploying Scan-to-HBIM V6 Cloud Functions
echo ==============================================
echo Project: %PROJECT_ID%
echo Region: %REGION%
echo ==============================================

REM Set project
call gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo [1/8] Enabling required APIs...
call gcloud services enable cloudfunctions.googleapis.com
call gcloud services enable cloudbuild.googleapis.com
call gcloud services enable storage.googleapis.com

REM Deploy Phase 1: Load
echo [2/8] Deploying phase-01-load...
cd phase_01_load
call gcloud functions deploy phase-01-load --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=%MEMORY% --timeout=%TIMEOUT%
cd ..

REM Deploy Phase 2: Preprocess
echo [3/8] Deploying phase-02-preprocess...
cd phase_02_preprocess
call gcloud functions deploy phase-02-preprocess --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=%MEMORY% --timeout=%TIMEOUT%
cd ..

REM Deploy Phase 3: Features
echo [4/8] Deploying phase-03-features...
cd phase_03_features
call gcloud functions deploy phase-03-features --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=4096MB --timeout=%TIMEOUT%
cd ..

REM Deploy Phase 4: Segment
echo [5/8] Deploying phase-04-segment...
cd phase_04_segment
call gcloud functions deploy phase-04-segment --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=%MEMORY% --timeout=%TIMEOUT%
cd ..

REM Deploy Phase 5: Classify
echo [6/8] Deploying phase-05-classify...
cd phase_05_classify
call gcloud functions deploy phase-05-classify --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=%MEMORY% --timeout=%TIMEOUT%
cd ..

REM Deploy Phase 6: Mesh
echo [7/8] Deploying phase-06-mesh...
cd phase_06_mesh
call gcloud functions deploy phase-06-mesh --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=4096MB --timeout=%TIMEOUT%
cd ..

REM Deploy Phase 7: IFC
echo [8/8] Deploying phase-07-ifc...
cd phase_07_ifc
call gcloud functions deploy phase-07-ifc --gen2 --runtime=%RUNTIME% --region=%REGION% --source=. --entry-point=process --trigger-http --allow-unauthenticated --memory=%MEMORY% --timeout=%TIMEOUT%
cd ..

echo.
echo ==============================================
echo DEPLOYMENT COMPLETE!
echo ==============================================
echo.
call gcloud functions list --filter="name~phase" --format="table(name,httpsTrigger.url)"
pause
