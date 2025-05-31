#1 download
python3 -m venv ./my_env #создать виртуальное окружение в папку 
. ./my_env/bin/activate   #активировать виртуальное ок
cd /var/lib/jenkins/workspace/MLops-Jenkins-Lab3   #перейти в директорию 
python3 -m ensurepip --upgrade
pip3 install setuptools
pip3 install -r requirements.txt    #установить пакеты python
python3 download.py    #запустить python script

#2 Train model
echo "Start train model"
cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate   #активировать виртуальное окружение
cd /var/lib/jenkins/workspace/MLops-Jenkins-Lab3 	   #перейти в директорию 
python3 train_model.py > best_model.txt #обучение модели запись лога в файл

#3 deploy
cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate   #активировать виртуальное окружение
cd /var/lib/jenkins/workspace/MLops-Jenkins-Lab3	   #перейти в директорию ./MLOPS/lab3
export BUILD_ID=dontKillMe            #параметры для jenkins чтобы не убивать фоновый процесс для mlflow сервиса
export JENKINS_NODE_COOKIE=dontKillMe #параметры для jenkins чтобы не убивать фоновый процесс для mlflow сервиса
path_model=$(cat /var/lib/jenkins/workspace/MLops-Jenkins-Lab3/best_model.txt) #прочитать путь из файла в bash переменную 
mlflow models serve -m $path_model -p 5003 --no-conda & 

#4 healthy
curl http://127.0.0.1:5003/invocations -H"Content-Type:application/json"  --data '{"inputs": [[ -1.75938045, -1.2340347 , -1.41327673,  0.76150439,  2.20097247, -0.10937195,  0.58931542,  0.1135538]]}'

# pipeline
pipeline {
    agent any

    stages {
        stage('Start Download') {
            steps {
                
                build job: "download"
                
            }
        }
        
        stage ('Train') {
            
            steps {
                
                script {
                    dir('/var/lib/jenkins/workspace/MLops-Jenkins-Lab3') {
                        build job: "Train model"
                    }
                }
            
            }
        }
        
        stage ('Deploy') {
            steps {
                build job: 'deploy'
            }
        }
        
        stage ('Status') {
            steps {
                build job: 'healthy'
            }
        }
    }
}
