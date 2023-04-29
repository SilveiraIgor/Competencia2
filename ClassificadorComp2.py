import numpy as np
from transformers import (BertTokenizerFast, BertForTokenClassification)
import torch
from sklearn import metrics

def TransformarTextoEmInput(textos):
    tokenizados = []
    for indice in range(len(textos)):
            tokens = tokenizer.encode_plus(textos[indice], add_special_tokens=True, truncation=True, return_tensors='pt')
            tokens = tokens['input_ids']
            #tokens = tokens.type(torch.int64).to(device)
            tokenizados.append(tokens)
    return tokenizados

def TransformarNotasEmVetor(textos, notas):
    novas_notas = []
    for indice in range(len(textos)):
            novas_notas.append(torch.tensor(notas[indice]).unsqueeze(0))
    return novas_notas

def le_dataset_argumentos(pasta):
    import os
    lista_argumentos = []
    lista_notas = []
    path = ".\\Argumentos-e-notas\\"+pasta+"\\"
    tamanho = len(os.listdir(path)) //2
    for i in range(tamanho):
        file_arg = open(path+str(i)+"A.txt","r",encoding='utf-8')
        file_nota = open(path+str(i)+"N.txt","r")
        lista_notas.append(float(file_nota.read()))
        args = file_arg.read()
        args = args.replace("\n"," [SEP] ")
        lista_argumentos.append(args)
        file_nota.close()
        file_arg.close()
    return lista_argumentos, lista_notas    


def filtrar_dataset(argumentos, notas, num_maximo):
    contador = [0]*6
    argumentos_filtrados = []
    notas_filtradas = []
    for arg, nota in zip(argumentos, notas):
        if contador[int(nota)] < num_maximo:
            argumentos_filtrados.append(arg)
            notas_filtradas.append(nota)
            contador[int(nota)] += 1
        #else:
            #print("filtrado")
    return argumentos_filtrados, notas_filtradas

def filtrar_dataset2(argumentos, notas, num_maximo, desejados):
    argumentos_filtrados = []
    notas_filtradas = []
    contador = [0]*2
    for arg, nota in zip(argumentos, notas):
        if int(nota) in desejados:
            argumentos_filtrados.append(arg)
            if nota == 4.0:
                notas_filtradas.append(1.0)
            else:
                notas_filtradas.append(0.0)
            contador[0] += 1
    return argumentos_filtrados, notas_filtradas

argumentos_treinamento, nota_treinamento = le_dataset_argumentos("Treinamento")
#argumentos_treinamento, nota_treinamento = filtrar_dataset2(argumentos_treinamento, nota_treinamento, 300, [0,4])
argumentos_validacao, nota_valid = le_dataset_argumentos("Validacao")
#argumentos_validacao, nota_valid = filtrar_dataset2(argumentos_validacao, nota_valid, 300, [0,4])

import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer


class CustomModel(nn.Module):
    def __init__(self): 
        super(CustomModel,self).__init__() 
        self.model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        #self.dropout = nn.Dropout(0.1) 
        self.classifier = nn.Sequential(nn.Linear(768,1020), nn.Tanh(), nn.Linear(1020,560), nn.Tanh(), nn.Linear(560,6), nn.Tanh(), nn.Linear(6,6), nn.Softmax(0))
        #self.classifier = nn.Sequential(nn.Linear(768,12), nn.Tanh(), nn.Linear(12,1))
    def forward(self, input_ids):
        
        #input_ids = input_ids.unsqueeze(0)
        #print("Input: ", input_ids.size())
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            outputs = outputs['pooler_output']
            #print(outputs.size())
        logits = self.classifier(outputs) 
        return logits
    
tokenizer_bertimbau = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',model_max_length=512, padding='max_length', truncation=True, do_lower_case=False)

def argumentos2inputs(lista_lista_argumentos):
    nova_lista = []
    for lista in lista_lista_argumentos:
        tokenizado = tokenizer_bertimbau(lista, padding='max_length')
        nova_lista.append(torch.tensor(tokenizado['input_ids']))
    print(len(nova_lista), len(lista_lista_argumentos))
    assert len(nova_lista) == len(lista_lista_argumentos)
    return nova_lista

def treinar(model, inputs, target):
    #loss_fn = torch.nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    #print(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    i = 0
    vetor_loss = []
    for batch, target in zip(inputs,target):
        #i += 1
        #print(i)
        batch = batch.to("cuda")
        #print("Target antes:", target)
        target = torch.tensor(target).long().to("cuda")
        optimizer.zero_grad()
        output = model(batch)
       # print("Output: ", output.size())
       # print("Target: ", target.size(), target)
        #output = output.squeeze(1)
        loss = loss_fn(output, target )
        vetor_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch.cpu().detach()
        target.cpu().detach()
    print("Loss media: ", sum(vetor_loss)/len(vetor_loss))

def acuracia_classe(respostas, gold_labels):
    classes_chutadas = [0]*6
    classes_acertadas = [0]*6
    classes_certas = [0]*6
    porcentagem_final = []
    for r, gl in zip(respostas, gold_labels):
        #print(gl)
        classes_certas[gl] += 1
        classes_chutadas[r] += 1
        if (r == gl):
            classes_acertadas[r] += 1
    for acertos, certo in zip(classes_acertadas, classes_certas):
        if certo != 0:
            porcentagem_final.append( float(acertos) / certo )
        else:
            porcentagem_final.append(0.0)       
    print("Chutei: ", classes_chutadas, sum(classes_chutadas))
    #print("Acertei: ", classes_acertadas, sum(classes_acertadas))
    print("Dist: ", classes_certas, sum(classes_certas))
    return porcentagem_final

    
#acuracia_classe([5, 5, 5, 5], [5, 5, 5, 5])
melhor_qwk = -1
melhor_epoca = -1
def testar(model, inputs, target, epoca):
    global melhor_qwk, melhor_epoca
    respostas = []
    target = np.array(target).flatten().astype(int)
    #print(target)
    with torch.no_grad():
        for batch in inputs:
            batch = batch.to("cuda")
            output = model(batch)
            nota = output.cpu().detach().numpy()
            #nota_final = np.rint(nota)
            nota_final = np.argmax(nota,axis=1)
            respostas.append(nota_final)
            batch.cpu().detach()
            #target.cpu().detach()
    respostas = np.array(respostas).flatten().astype(int)
    #print(respostas.size, target.size )
    #print(target)
    QWK = metrics.cohen_kappa_score(target, respostas, weights='quadratic')
    print("QWK: ", QWK)
    #print("RMSE: ", metrics.mean_squared_error(target, respostas, squared=False))
    print("MSE: ", metrics.mean_squared_error(target, respostas, squared=True))
    print("Acc: ", metrics.accuracy_score(target, respostas))
    print("Porcentagem das classes: ", acuracia_classe(respostas, target))
    if (QWK > melhor_qwk):
        melhor_qwk = QWK
        melhor_epoca = epoca
    print(f"Estou na epoca {epoca} e a melhor QWK foi {melhor_qwk} na epoca {melhor_epoca}")

def criar_batch_size(lista_tensores, lista_notas, tamanho):
    inicio = 0
    fim = tamanho
    nova_lista = []
    novas_notas = []
    while( (inicio < len(lista_tensores)) and (fim < len(lista_tensores))):
        stack = lista_tensores[inicio:fim]
        notas = lista_notas[inicio:fim]
        nova_lista.append(torch.stack(stack))
        novas_notas.append(notas)
        inicio += tamanho
        fim += tamanho
    return nova_lista, novas_notas


torch.cuda.empty_cache()
t = argumentos2inputs(argumentos_treinamento)
t2 = argumentos2inputs(argumentos_validacao)
batch_size = 8
t, nota_treinamentoN = criar_batch_size(t, nota_treinamento, batch_size)
t2, nota_validN = criar_batch_size(t2, nota_valid, batch_size)
t2 = torch.stack(t2)
print(len(t))

model2 = CustomModel().to("cuda")
for i in range(30):
    print("Epoca: ", i+1)
    print("  Treinamento:")
    treinar(model2, t, nota_treinamentoN)
    #print("   Teste no treinamento:")
    #testar(model2, t, nota_treinamentoN, i)
    print("  Validacao:")
    testar(model2, t2, nota_validN, i)