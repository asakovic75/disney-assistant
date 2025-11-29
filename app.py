import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
body, .stApp { font-family: 'Nunito', sans-serif !important; }
h1 { font-size: 1.8rem !important; color: #1f77b4; }
h3 { font-size: 1.2rem !important; }
.stButton button {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 3em;
}
.answer-card {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.reasoning-box {
    font-size: 0.85rem;
    color: #6c757d;
    border-left: 3px solid #ffc107;
    padding-left: 10px;
    margin-bottom: 20px;
}
.section-header {
    color: #2c3e50;
    font-weight: bold;
    margin-top: 20px;
    border-bottom: 2px solid #17a2b8;
    display: inline-block;
    padding-bottom: 5px;
}
"""
st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def create_knowledge_base():
    try:
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            knowledge_base += f"Название: {work['Name']}\n"
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n"
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска','не указано')}\n"
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы','не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', 'не указано')}\n"
            knowledge_base += f"Исполнители: {work.get('Исполнители','не указано')}\n"
            knowledge_base += f"Персонажи: {work.get('Персонажи', 'не указано')}\n"
            knowledge_base += f"Описание: {work.get('Описание', 'Нет описания')}\n"
        return knowledge_base
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None

st.title("✨ Умный ассистент Пиксель")

col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        label=" ",
        placeholder="Спросите что-нибудь о произведениях Disney (например: 'Какие есть фильмы про принцесс?')",
        label_visibility="collapsed"
    )
with col2:
    ask_button = st.button("Найти", use_container_width=True)

knowledge_base_text = create_knowledge_base()
answer_placeholder = st.empty()

if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "meta-llama/llama-3.3-70b-versatile"
    except Exception as e:
        st.error(f"Ошибка инициализации клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Анализирую архивы Disney..."):
            try:
                system_prompt = f"""
                Ты - Пиксель, эксперт по базе данных Disney.
                
                ТВОЯ ЗАДАЧА:
                1. Проанализировать запрос пользователя и предоставленные ДАННЫЕ.
                2. Сначала выполнить шаг [РАССУЖДЕНИЯ]: найти все подходящие карточки, проверить их поля.
                3. Сформировать шаг [ОТВЕТ]: 
                   - Если найдены произведения, ОБЯЗАТЕЛЬНО раздели их на две группы: "🎬 Фильмы" и "🦄 Мультфильмы/Анимация" (используй поле 'Тип').
                   - Для каждого произведения создай краткую красивую HTML-карточку.
                   - Если информации нет, ответь: "К сожалению, эта информация не найдена в архиве."
                
                СТРОГИЕ ПРАВИЛА:
                - НИКАКИХ ДОГАДОК. Используй ТОЛЬКО текст из раздела ДАННЫЕ ниже.
                - Ответ должен быть на русском языке.
                - Используй HTML теги для форматирования ответа (<b>, <i>, <br>, <ul>, <li>).
                
                ДАННЫЕ:
                {knowledge_base_text}
                """

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Вопрос: {user_query}. В ответе используй формат: [РАССУЖДЕНИЯ] ... [ОТВЕТ] ..."}
                    ],
                    temperature=0.1,
                    max_tokens=2500
                )
                
                full_text = response.choices[0].message.content
                
                try:
                    parts = full_text.split("[ОТВЕТ]")
                    reasoning = parts[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer = parts[1].strip() if len(parts) > 1 else ""
                except IndexError:
                    reasoning = ""
                    final_answer = full_text.replace("[РАССУЖДЕНИЯ]", "").replace("[ОТВЕТ]", "")

                with st.expander("🔍 Анализ и поиск (Рассуждения ИИ)"):
                    st.markdown(f"<div class='reasoning-box'>{reasoning}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='answer-card'>{final_answer}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Произошла ошибка при обработке запроса: {e}")

elif not GROQ_API_KEY:
    st.warning("Пожалуйста, укажите GROQ_API_KEY в переменных окружения.")
