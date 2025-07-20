This is the updated version of the outdated course "https://www.youtube.com/watch?v=-dJPoLm_gtE". But dont worry o gto you..... I am still working on the deployment part(My suggestion go with GCP or AWS). That said,
the issue starts from tracking the pipline with MLFlow. For that, use these commands in the terminal. Make sure to download all the the necessary packages.

There are three different terms MLFlow tracker, MLFlow stack and how to register each one of these. (Just ChatGPT it!)

Creation:
"zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri="127.0.0.1:5000" --tracking_username=default --tracking-password=default"     (Use this to create a new experiment tracker)

"zenml stack register mlflow_stack --orchestrator-default --artifact=default --experiment-tracker=mflow_tracker"  (Use this to create a new stack)


Set this stack to use it

"zenml stack set mlflow_stack"  

"zenml stack describe"

That is all!! Dont forget to star the repo. Thanks in advance!!!

