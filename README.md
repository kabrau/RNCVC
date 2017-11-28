**Redes Neurais Convolucionais para Visão Computacional**  
#Trabalho 2: Redes Neurais Convolucionais

### 1) Enunciado
No segundo trabalho da disciplina, você deverá utilizar Redes Neurais Convolucionais (ConvNets) na resolução de algum problema. O trabalho deverá ser realizado em trios. Cada grupo é livre para escolher o problema, o framework, a arquitetura da rede e os hiperparâmetros. Os grupos deverão entregar um artigo de até 4 páginas (seguindo o [template IEEE de Conferências](https://www.ieee.org/conferences_events/conferences/publishing/templates.html)) que descreva o trabalho realizado (uma quinta página pode conter apenas referências bibliográficas). As partes obrigatórias do artigo são: resumo, introdução, método, resultados, conclusão e referencias. Na falta de espaço, pode-se enviar ﬁguras, tabelas, gráficos ou vídeos como material suplementar ou apêndice. O artigo poderá ser escrito em português ou em inglês.

### 2) Notas adicionais 
 - Você não precisa utilizar redes extremamente profundas ou bases de dados muito grandes. 
 - Você pode adotar estrategias como ﬁne-tuning para reduzir o tempo necessário para treinamento. 
 - Em caso de necessidade, cada grupo poderá agendar uma data para utilização de uma GPU de alto desempenho do GPIN.

### 3) Objetivos do trabalho
- Compreender o protocolo completo de treinamento, validação e avaliação de ConvNets. 
- Compreender o impacto de diferentes escolhas de arquiteturas e hiperparâmetros no processo de treinamento. 
- Compreender as possibilidades existentes no uso das informações obtidas através de ConvNets. 
- Ter experiencia com um framework de Deep Learning. 
- Executar experimentos de acordo com uma metodologia. 
- Avaliar os resultados obtidos.


## INSTALL  
1- Baixe e Instale o conda: https://www.anaconda.com/download/  
2- \> conda create -n tensorflow python=3.5  
3- \> activate tensorflow  
4- (tensorflow) \> pip install --ignore-installed --upgrade tensorflow  ( or tensorflow-gpu )  
5- (tensorflow) \> pip install keras
6- (tensorflow) \> pip install opencv-python
7- (tensorflow) \> pip install sklearn
8- (tensorflow) \> pip install h5py

## Dataset prepar
1- Download and extract dataset, pedestrian_signal_classification.zip, https://drive.google.com/open?id=0B2MY7T7S8OmJVVlCTW1jYWxqUVE
2- create the dataset.py file
2- create get_best_size(dir) to show the mean size images
2- create load_data(dir, h, w) to load (and resize) images and create labels

## PIPELINE
1- Install framework
2- Download train example cifar10: https://raw.githubusercontent.com/fchollet/keras/master/examples/cifar10_cnn.py to train.py
3- run train cifar10 (just for test, do not really train)
4- create dataset prepar
5- change train.py to use load_data(...)
6- option to train with ou without data_augmentation
7- change train.py to show tensorBoard: tensorboard --logdir=./logs
8- change train.py to save best model (ModelCheckpoint)

