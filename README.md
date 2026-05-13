# Segmentação Semântica em Ambientes Urbanos

Pipeline completo de segmentação semântica aplicado ao dataset **CamVid**,
utilizando U-Net com encoder ResNet-34 pré-treinado no ImageNet.

---

## A História do Projeto

Este repositório tem duas versões — e a diferença entre elas é parte
intencional da narrativa.

### Versão 2025 — Aprendendo do Zero

O projeto nasceu como trabalho em grupo durante um bootcamp de dados.
O programa cobriu Python, manipulação e visualização de dados e
introdução ao aprendizado de máquina. Foi uma base sólida, mas que
naturalmente não aprofundou os detalhes práticos de como treinar modelos
de visão computacional. A parte de segmentação semântica foi
construída explorando notebooks da comunidade no Kaggle, adaptando
o que encontrávamos e aprendendo na prática.

Chegamos a um notebook funcional, mas com problemas sérios que só
identificamos depois com os conceitos mais aprofundados:

- **Métricas completamente quebradas** — IoU retornava `-1.12`, um valor
  impossível. A implementação usava `smp.utils.metrics` de forma incorreta
  para segmentação multiclasse
- **"Focal Loss" era Cross-Entropy** — o nome estava errado, a matemática
  estava errada
- **32 classes incluindo Train e Tunnel** com zero imagens no dataset
- **Sem análise de desequilíbrio** — não sabíamos que Road ocupava 28% dos
  pixels e SignSymbol apenas 0.11%
- **Paths hardcoded** (`C:/Users/Fileto/...`) — o notebook não rodava em
  nenhuma outra máquina
- **Classe Void incluída no treino** — pixels sem anotação distorcendo
  o aprendizado sem que percebêssemos

O modelo pode ter aprendido algo útil — nunca saberemos, porque nunca
conseguimos medi-lo corretamente.

### Versão 2026 — Refeito com Fundamento

Com os conceitos mais aprofundados, o projeto foi reconstruído do zero.
Cada decisão tem justificativa técnica, cada número pode ser explicado.

| | Versão 2025 | Versão 2026 |
|---|---|---|
| **IoU** | -1.12 ❌ (quebrado) | 0.506 ✅ (real) |
| **Métricas** | smp.utils (incorreto) | torchmetrics (correto) |
| **Loss** | "Focal Loss" = Cross-Entropy | Dice + CE com pesos de classe |
| **Classes** | 32 (incluindo Train=0) | 21 (critério analítico) |
| **Análise de dados** | Contagem de imagens | Distribuição pixel-level |
| **Desequilíbrio** | Ignorado | Pesos de classe calculados |
| **Void** | Incluído no treino | ignore_index |
| **Duplicatas** | Ignoradas | Detectadas e removidas |
| **Paths** | Hardcoded | Relativos e reproduzíveis |
| **Reprodutibilidade** | Nenhuma | SEED fixo, requirements.txt |

---

## Dataset

**CamVid** — Cambridge-driving Labeled Video Database  
Sequências de vídeo urbano capturadas em Cambridge, UK, com anotações
pixel-level de 32 classes.

📦 [Download no Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid)

Após análise de distribuição pixel-level, 21 classes foram mantidas
para treinamento. Classes com menos de 0.1% dos pixels totais foram
removidas por insuficiência de representação. A classe Void
(regiões sem anotação) é ignorada durante treino e avaliação.

| Split | Imagens |
|---|---|
| Treino | 347 (após remoção de duplicatas) |
| Validação | 101 |
| Teste | 233 |

---

## Arquitetura

**U-Net** com encoder **ResNet-34** pré-treinado no ImageNet.

- O encoder extrai características progressivamente abstratas da imagem
- O decoder reconstrói a resolução original via *skip connections*
- Transfer Learning do ImageNet acelera a convergência e melhora resultados

---

## Resultados

Avaliação no conjunto de teste após 30 épocas de treinamento:

| Métrica | Valor |
|---|---|
| **IoU Macro** | **0.5061** |
| F1 / Dice | 0.6393 |
| Accuracy | 0.7221 |

### IoU por Classe (destaques)

| Classe | IoU | | Classe | IoU |
|---|---|---|---|---|
| Sky | 0.92 ✅ | | Column_Pole | 0.19 ⚠️ |
| Road | 0.87 ✅ | | SignSymbol | 0.19 ⚠️ |
| Building | 0.78 ✅ | | Truck_Bus | 0.12 ⚠️ |
| Car | 0.74 ✅ | | | |

Classes grandes e visualmente distintas são segmentadas com excelência.
Objetos finos (postes, sinais) e subtipos de veículos permanecem como
desafios — comportamento esperado para esta arquitetura e escala de dados.

---

## Como Rodar

### Requisitos

- Python 3.12
- GPU com suporte CUDA (treinamento)

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/segmentacao-ambientes-naturais
cd segmentacao-ambientes-naturais

# 2. Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instale o PyTorch (gere o comando para sua GPU em pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Instale as demais dependências
pip install -r requirements.txt
```

### Estrutura de Pastas

```
segmentacao-ambientes-naturais/
├── Segmentacao_ambientes_naturais.ipynb
├── requirements.txt
├── README.md
└── camvid/               ← baixar do Kaggle (não versionado)
    └── CamVid/
        ├── train/
        ├── train_labels/
        ├── val/
        ├── val_labels/
        ├── test/
        ├── test_labels/
        └── class_dict.csv
```

### Execução

Abra o notebook no VS Code ou Jupyter e execute as células em ordem.
A célula de configuração central (`CONFIGURAÇÃO CENTRAL`) é o único
lugar que precisa ser ajustado caso a estrutura de pastas seja diferente.

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| PyTorch | Framework de deep learning |
| segmentation-models-pytorch | Arquitetura U-Net |
| torchmetrics | Métricas corretas para multiclasse |
| albumentations | Augmentação de dados |
| OpenCV + Pillow | Manipulação de imagens |
| pandas + numpy | Análise de dados |
| matplotlib + seaborn | Visualização |

---

## Próximos Passos (v3)

- Arquitetura **DeepLabV3+** com encoder **EfficientNet-B4**
- Augmentações para baixa iluminação (`RandomGamma`, `CLAHE`)
- **OHEM** — foco do treino nos exemplos mais difíceis
- **Cosine Annealing** com warm restarts
- Meta: IoU ≥ 0.62

