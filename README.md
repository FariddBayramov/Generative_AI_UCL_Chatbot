# 🏆UEFA Champions League Chatbot

This project involves the development of an AI-powered chatbot that can answer users' questions about the UEFA Champions League. The chatbot generates the appropriate response by classifying the user's intention (intent classification) and focuses only on Champions League related topics.

## 🎯 Project Objective

- Analysing the user's questions to determine their intentions.
- Developing a chatbot that only provides information about the UEFA Champions League.
- Generating the output of the chatbot using different large language models (LLM).
- To compare the performance of the models used.

---

## 🚀 Run Instructions

1. **Install the requirements:**

```bash
pip install -r requirements.txt
```
2. **Add your API keys to the .env file:**
```bash
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```
3. **Start the Streamlit app:**
```bash
streamlit run app.py
```

## 🧠 Chatbot Flow

The chatbot can respond to the following core intents:

- `Greeting`: Greetings (e.g., Hello)
- `Goodbye`:  Farewells (e.g., See you)
- `Reject`: Rejection
- `non-champions`: Detection of questions unrelated to the Champions League
- `Info_*`: Specific scenarios about the Champions League (e.g., Info_Players, Info_Records, etc.)
- `History`: Historical information about the tournament and teams
- And a total of `45` different intent types, including many others...

### Chatbot Flow Description

1. Text input is received from the user.
2. The text is vectorized using `TF-IDF`, and intent classification is performed with `SVM`.
3. If the intent is not related to the Champions League, the chatbot politely indicates this.
4. If the intent is related to the Champions League, content is selected using a vector-based retrieval system` (RAG)`.
5. The identified intent and retrieved information are sent as a prompt to the selected LLM.
6. The response from the AI model is displayed to the user.
7. During response generation, a ChatOpenAI call is made using the Langchain library. The predicted intent is included first in the prompt, followed by the vector-based information. This ensures that the answers remain on-topic and contextually relevant.
---

## 🗃️ Dataset

The dataset is in `.xlsx` format and is located at `data/champions_league_chatbot_dataset.xlsx`. The dataset is structured in the following format:

| Intent          |                 Example                  |     
|-----------------|------------------------------------------|
| Greeting        | Nice to see you                          | 
| Goodbye         | Until we meet again                      | 
| Tournament_info | Tell me about the champions league.?     |     
|       ...       |             ...                          |                         
|       ...       |             ...                          |                        
|       ...       |             ...                          |           

- Contains a total of 45 intent types with **1195** example sentences.
- Dataset Kaggle link: https://www.kaggle.com/datasets/feridbayramov/champions-league-chatbot-dataset
---

## 🤖  Models Used (LLM)

Two different LLMs are used in the project:

| Model Name                                 |        Provider      |
|--------------------------------------------|----------------------|
| `GPT-3.5-Turbo`                            |         OPENAI       |
| `qwen/qwen3-235b-a22b:free`                | Alibaba (OpenRouter) |

### Reason for Model Selection

- **GPT-3.5 Turbo:** Fast and cost-effective. Powerful for general-purpose dialogues and has a broad knowledge base.
- **Qwen-3 235B:** With its large capacity and dialogue capabilities, it can generate detailed and accurate responses and is free to use.


### APIs and Tools Used

- API keys obtained via [Platform OPENAI](https://platform.openai.com/api-keys) and [OpenRouter.ai](https://openrouter.ai).
- Integration with OpenRouter API done using the `openai` Python package.
- API keys are read from the `.env` file.


## 🧩 Intent Classifier

- In this project, a `TF-IDF + SVM` based intent classifier is used to understand user queries.

- `TfidfVectorizer`: Creates word vectors using n-grams (1,2).

- `SVC (Support Vector Machine)`: Performs classification with a linear kernel.

- After training, the model is saved as `models/intent_classifier.pkl`.

- Real-time predictions can be made using the `predict_intent()` function.

- Train/Test split ratio: **70% / 30%**

⚠️ During training, a maximum feature frequency (`max_df=0.95`) was applied and English stopwords were filtered.

![intent](https://github.com/user-attachments/assets/076163ad-fe7a-4025-b132-16745e2893ed)

## 🔍 Vector-Based Information Retrieval (RAG)

The chatbot’s knowledge source is only the `data/champions_league_information.txt` file. This text file is split into small chunks using `CharacterTextSplitter` and vectorized with `OpenAIEmbeddings`, then added to the `Chroma` vector database.

**Configuration used:**

- `CharacterTextSplitter`: `chunk_size=500`, `chunk_overlap=50`  
- `OpenAIEmbeddings`: `"text-embedding-3-large"` model  
- `Chroma`: Persistent storage folder `"chroma_db"`  

**Information Retrieval Process:**

User question → sent to intent classifier.  
Classified intent → retriever fetches similar paragraphs.  
Retrieved information + user question → sent as prompt to the LLM.

## 📊 Model Performance Comparison

For intent classification, a `TfidfVectorizer + SVC` pipeline was used. Evaluation metrics include:

- **Precision**  
- **Recall**  
- **F1 Score**  
- **Confusion Matrix**  

### Train/Test Split

- Training data: 70%  
- Test data: 30%  
- All models were evaluated on the same test set.

### Performance Results

| intent               |   precision |   recall |   f1_score |   support |
|:---------------------|------------:|---------:|-----------:|----------:|
| VAR_info             |        1.00 |     1.00 |       1.00 |         7 |
| anthem_info          |        0.71 |     1.00 |       0.83 |         5 |
| assist_leaders       |        1.00 |     1.00 |       1.00 |        12 |
| ball_info            |        1.00 |     1.00 |       1.00 |         9 |
| best_players         |        1.00 |     1.00 |       1.00 |         6 |
| broadcast_info       |        1.00 |     1.00 |       1.00 |         9 |
| champions_history    |        1.00 |     1.00 |       1.00 |         7 |
| clean_sheets         |        0.83 |     1.00 |       0.91 |         5 |
| club_stats           |        1.00 |     0.86 |       0.92 |         7 |
| coach_info           |        1.00 |     1.00 |       1.00 |         9 |
| draw_info            |        1.00 |     1.00 |       1.00 |         5 |
| fan_chants           |        1.00 |     1.00 |       1.00 |         6 |
| final_info           |        1.00 |     1.00 |       1.00 |         7 |
| format_info          |        0.75 |     0.75 |       0.75 |         4 |
| fun_facts            |        1.00 |     0.78 |       0.88 |         9 |
| goodbye              |        1.00 |     0.78 |       0.88 |         9 |
| greeting             |        1.00 |     0.83 |       0.91 |         6 |
| group_standings      |        1.00 |     1.00 |       1.00 |         8 |
| hat_tricks           |        1.00 |     1.00 |       1.00 |        10 |
| history              |        1.00 |     0.73 |       0.84 |        11 |
| injury_news          |        1.00 |     1.00 |       1.00 |         5 |
| knockout_stage       |        1.00 |     1.00 |       1.00 |         9 |
| language_support     |        1.00 |     1.00 |       1.00 |         7 |
| match_result         |        1.00 |     1.00 |       1.00 |         7 |
| match_schedule       |        1.00 |     1.00 |       1.00 |         8 |
| multiple_titles      |        1.00 |     0.89 |       0.94 |         9 |
| non-champions_league |        0.91 |     1.00 |       0.95 |        29 |
| penalty_info         |        1.00 |     1.00 |       1.00 |        13 |
| player_info          |        1.00 |     1.00 |       1.00 |         5 |
| prediction           |        1.00 |     1.00 |       1.00 |         7 |
| ranking              |        1.00 |     1.00 |       1.00 |         6 |
| record_wins          |        1.00 |     1.00 |       1.00 |         7 |
| red_cards            |        0.75 |     1.00 |       0.86 |         6 |
| referee_info         |        1.00 |     1.00 |       1.00 |        10 |
| rules                |        1.00 |     0.88 |       0.93 |         8 |
| sponsorship          |        1.00 |     1.00 |       1.00 |        10 |
| suspensions          |        1.00 |     1.00 |       1.00 |         4 |
| team_info            |        1.00 |     1.00 |       1.00 |         7 |
| team_titles          |        0.80 |     1.00 |       0.89 |         8 |
| tickets_info         |        1.00 |     1.00 |       1.00 |         8 |
| tournament_info      |        0.90 |     0.90 |       0.90 |        10 |
| travel_info          |        1.00 |     1.00 |       1.00 |         9 |
| trophy_info          |        0.89 |     1.00 |       0.94 |         8 |
| venue_info           |        1.00 |     1.00 |       1.00 |         8 |
|`accuracy`          |             |          |       **0.96** |       **359** |          
| `macro avg`            |        **0.97** |     **0.96** |       **0.96** |       **359** |
| `weighted avg`         |        **0.97** |     **0.96** |       **0.96** |       **359** |

---



**Confusion Matrix:**

![conf](https://github.com/user-attachments/assets/659c0478-ca66-4262-a58c-be1d590ddef6)

---

## 💬 Application Interface

- An advanced interface is provided using `streamlit` for a visual UI.  
- Text input is received from the user and the response is displayed on the screen.

![gui](https://github.com/user-attachments/assets/a5207a89-6b9e-4244-bda6-67fcae663ef5)

---

## 📁 Project Structure
```
├── data/
│   ├── champions_league_information.txt        # Text file with information about the Champions League
│   └── champions_league_chatbot_dataset.xlsx   # Intent classification dataset
├── models/
│   └── intent_classifier.pkl                    # Trained TF-IDF + SVM model
├── app/
│   ├── streamlit_app.py                         # Streamlit application for the user interface
│   ├── gpt_model.py
│   ├── qwen_model.py
├── README.md                                    # Project documentation
└── requirements.txt                             # Required Python libraries

