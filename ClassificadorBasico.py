from build_dataset import Corpus
import numpy as np
from sklearn import metrics
import sklearn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoTokenizer
from transformers import AutoModel

class Dataset:
    def __init__(self, competence):
        self.c = Corpus()
        self.c.read_corpus().shape
        self.train, self.valid, self.test = self.c.read_splits()
        #print(self.train.shape)
        self.train.loc[1:5, ['essay', 'score', 'competence']] 
        #print(self.valid.shape)
        self.valid.loc[1:5, ['essay', 'score', 'competence']]
        #print(self.test.shape)
        self.test.loc[1:5, ['essay', 'score', 'competence']]
        self.competence = competence
    def UnirListas(self, lista):
        unificado = ""
        unificado = unificado+lista[0]+'\n'
        for t in range(1,len(lista)):
            unificado += lista[t]+'\n'
        return unificado
        
    def gerarTreinamento(self):
        """
        retorna um conjunto (texto, nota_competencia1). Um texto eh composto de varios paragrafos, cada paragrafo eh uma lista 
        """
        textos = []
        notas = []
        for index, row in self.train.iterrows():
            texto = self.UnirListas(row['essay'])
            notas.append( float(row['competence'][self.competence] / 40))
            textos.append(texto)
        return textos, notas
    def gerarTeste(self):
        textos = []
        notas = []
        for index, row in self.test.iterrows():
            texto = self.UnirListas(row['essay'])
            notas.append( float(row['competence'][self.competence] / 40))
            textos.append(texto)
        return textos, notas
    def gerarValidacao(self):
        textos = []
        notas = []
        for index, row in self.valid.iterrows():
            texto = self.UnirListas(row['essay'])
            notas.append( float(row['competence'][self.competence] / 40))
            textos.append(texto)
        return textos, notas

def TransformarTextoEmInput(textos):
    tokenizados = []
    for indice in range(len(textos)):
            tokens = tokenizer.encode_plus(textos[indice], add_special_tokens=True, truncation=True, return_tensors='pt')
            tokens = tokens['input_ids']
            tokens = tokens.type(torch.long).to(device)
            tokenizados.append(tokens)
    return tokenizados

def TransformarNotasEmVetor(textos, notas):
    novas_notas = []
    for indice in range(len(textos)):
            novas_notas.append(torch.tensor(int(notas[indice]), dtype=torch.long).unsqueeze(0).to(device))
    return novas_notas



import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self): 
        super(CustomModel,self).__init__() 
        self.model = AutoModel.from_pretrained(modelo).to(device)
        self.classifier = nn.Sequential(nn.Linear(768,6), nn.Softmax(dim=0))
    def forward(self, input_ids):
        #print("Input: ", input_ids.size())
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            #outputs = torch.squeeze(outputs['pooler_output'])
            outputs = outputs.last_hidden_state[0][0]
        logits = self.classifier(outputs) 
        return logits

def treinar(model, inputs, target):
    #loss_fn = torch.nn.MSELoss()
    #print(pesos)
    loss_fn = nn.CrossEntropyLoss(weight=pesos)
    #loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    vetor_loss = []
    optimizer.zero_grad()
    count = 0
    for index in range(len(inputs)):
        count += 1
        output = model(inputs[index])
        output = output.unsqueeze(0)
        #r = torch.tensor(target[index]).unsqueeze(0).long()
        #print(target[index])
        #r = torch.tensor(target[index], dtype=torch.long)
        r = target[index]
        #print(r)
        #print(output, r)
        loss = loss_fn(output, r)
        vetor_loss.append(loss.item())
        loss.backward()
        if (count == 1):    
            optimizer.step()
            optimizer.zero_grad()
            count = 0
    loss_tmp = sum(vetor_loss)/len(vetor_loss)
    print("Loss media: ", loss_tmp)
    
def TokenizarUmParagrafo(paragrafo):
    tokens = tokenizer.encode_plus(paragrafo, add_special_tokens=True, truncation=True, return_tensors='pt')
    tokens = tokens['input_ids']
    tokens = tokens.type(torch.int64).to(device)
    return tokens

def acuracia_classe(respostas, gold_labels):
    classes_chutadas = [0]*6
    classes_acertadas = [0]*6
    classes_certas = [0]*6
    porcentagem_final = []
    for r, gl in zip(respostas, gold_labels):
        #print(gl)
        classes_certas[int(gl)] += 1
        classes_chutadas[int(r)] += 1
        if (r == gl):
            classes_acertadas[int(r)] += 1
    for acertos, certo in zip(classes_acertadas, classes_chutadas):
        if certo != 0:
            porcentagem_final.append( float(acertos) / certo )
        else:
            porcentagem_final.append(0.0)       
    print("Chutei: ", classes_chutadas, sum(classes_chutadas))
    print("Acertei: ", classes_acertadas, sum(classes_acertadas))
    print("Dist: ", classes_certas, sum(classes_certas))
    return porcentagem_final

maior_qwk = -2
def testar(model, inputs, target, tipo):
    global maior_qwk
    respostas = []
    for index in range(len(inputs)):
        #notas_parciais = []
        with torch.no_grad():
            tokenizado = TokenizarUmParagrafo(inputs[index])
            output = model(tokenizado)
            nota = output.cpu().detach().numpy()
            #notas_parciais.append(nota)
            #print(len(notas_parciais))
            #nota_final = np.rint(np.average(notas_parciais))
            nota_final = np.argmax(nota)
        respostas.append(nota_final)
    #print(respostas)
    QWK = metrics.cohen_kappa_score(target, respostas, weights='quadratic')
    if(tipo=="validacao"):
        if (QWK > maior_qwk):
            print("Encontrei uma QWK maior <<<<<")
            maior_qwk = QWK
    print("QWK: ", QWK)
    #print("RMSE: ", metrics.mean_squared_error(target, respostas, squared=False))
    print("MSE: ", metrics.mean_squared_error(target, respostas, squared=True))
    print("Porcentagem das classes: ", acuracia_classe(respostas, target))
    print("Total acc: ", metrics.accuracy_score(target, respostas))


#modelo = 'xlm-roberta-large'
modelo = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo,model_max_length=512, truncation=True, do_lower_case=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
ds = Dataset(1)
texto_treinamento, nota_treinamento = ds.gerarTreinamento()
texto_teste, nota_teste = ds.gerarTeste()
texto_valid, nota_valid = ds.gerarValidacao()
novos_inputs = TransformarTextoEmInput(texto_treinamento)
print(len(novos_inputs))
novas_notas = TransformarNotasEmVetor(texto_treinamento, nota_treinamento)
print(len(novas_notas))
model2 = CustomModel().to(device)
pesos = sklearn.utils.class_weight.compute_class_weight('balanced', classes=[0.0,1.0,2.0,3.0,4.0,5.0], y=nota_treinamento)
pesos = torch.tensor(pesos).float().to("cuda")
print(pesos)

for i in range(5):
    print("Iteracao ", i+1)
    treinar(model2, novos_inputs, novas_notas)
    print("-- Treinamento: ")
    testar(model2, texto_treinamento, nota_treinamento, "treinamento")
    print("--Validacao:")
    testar(model2, texto_valid, nota_valid, "validacao")
    print("--Teste:")
    testar(model2, texto_teste, nota_teste, "teste")