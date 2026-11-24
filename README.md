# Reconhecimento Facial em Tempo Real com DeepFace e MySQL

Sistema de visão computacional em Python para detecção e identificação biométrica facial em tempo real via webcam, com logging e persistência em banco de dados relacional (MySQL).

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Biometria & IA:** [DeepFace](https://github.com/serengil/deepface) (Modelo `VGG-Face`, detector `OpenCV`, distância `Cosine`)
- **Visão Computacional:** OpenCV (`cv2`)
- **Persistência:** MySQL Connector (`mysql-connector-python`)
- **Concorrência:** Multi-threading assíncrono com `Queue` para processamento paralelo sem travar a taxa de quadros (FPS) da câmera

---

## 📂 Estrutura do Projeto

```text
reconhecimento-facial/
├── database/                   # Base de imagens para treinamento/comparação
│   └── pessoa_exemplo/         # Cada subpasta representa uma pessoa
│       └── .gitkeep
├── .env.example                # Modelo de variáveis de ambiente
├── .gitignore                  # Arquivos ignorados (fotos pessoais, venvs, DBs)
├── main.py                     # Script principal de captura, reconhecimento e persistência
└── README.md                   # Documentação do projeto
```

---

## 🚀 Como Executar

### 1. Pré-requisitos

Instale os pacotes necessários:
```bash
pip install deepface opencv-python mysql-connector-python numpy
```

### 2. Configurar o Banco de Dados

Crie o banco de dados MySQL e a tabela para registrar os reconhecimentos:
```sql
CREATE DATABASE face_database;
USE face_database;

CREATE TABLE face_reconhecidas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    confianca FLOAT NOT NULL,
    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Configure as credenciais copiando o `.env.example`:
```bash
cp .env.example .env
```
Ou defina as variáveis no seu ambiente:
- `MYSQL_HOST` (padrão: `localhost`)
- `MYSQL_USER` (padrão: `face`)
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE` (padrão: `face_database`)

### 3. Cadastrar Faces

Crie uma pasta com o nome da pessoa dentro do diretório `database/` e adicione fotos nítidas do rosto:
```text
database/
├── joao/
│   ├── foto1.jpg
│   └── foto2.jpg
└── maria/
    ├── foto1.jpg
    └── foto2.jpg
```

### 4. Iniciar o Reconhecimento

Execute o script principal:
```bash
python main.py
```

- Pressione **`q`** ou **`ESC`** para fechar a janela de exibição.
- O sistema exibirá uma caixa delimitadora **verde** para rostos identificados e **vermelha** para rostos desconhecidos.
- Identificações com confiança acima do limiar configurado (`RECOGNITION_THRESHOLD`) são salvas automaticamente no MySQL em intervalos de controle.
