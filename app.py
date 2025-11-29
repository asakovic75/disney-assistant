import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
body, .stApp { font-family: 'Nunito', sans-serif !important; }
h1, h2, h3 { color: #0e1117; }
.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 20px;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
}
.stTextInput input {
    border-radius: 20px;
}
.answer-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.movie-tag {
    background-color: #e6f3ff;
    color: #0066cc;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    font-weight: bold;
}
.cartoon-tag {
    background-color: #fff0f5;
    color: #cc0066;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    font-weight: bold;
}
.reasoning-box {
    background-color: #f8f9fa;
    border-left: 4px solid #ffc107;
    padding: 15px;
    margin-bottom: 20px;
    font-size: 0.9em;
    color: #555;
    border-radius: 0 10px 10px 0;
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
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы','не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска','не указано')}\n"
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Исполнители: {work.get('Исполнители','не указано')}\n"
            knowledge_base += f"Награды: {work.get('Награды', 'не указано')}\n"
            knowledge_base += f"Персонажи: {work.get('Персонажи', 'не указано')}\n"
            knowledge_base += f"Песни: {work.get('Песни', 'не указано')}\n"
            knowledge_base += f"Продолжительность: {work.get('Продолжительность', 'не указано')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', 'не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n"
        return knowledge_base
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None

st.markdown("### ✨ Умный ассистент Пиксель")

col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        label=" ",
        placeholder="Спросите что-нибудь о произведениях Disney (например: 'Фильмы про принцесс')...",
        key="user_input_box",
        label_visibility="collapsed"
    )
with col2:
    ask_button = st.button("Найти", use_container_width=True, key="find_answer")

knowledge_base_text = create_knowledge_base()
answer_placeholder = st.empty()

if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "llama-3.1-70b-versatile"
    except Exception as e:
        st.error(f"Ошибка инициализации клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Пиксель ищет информацию..."):
            try:
                prompt = f"""
                Твоя роль - быть ассистентом-базой данных Disney "Пиксель".
                
                КОНТЕКСТ (БАЗА ДАННЫХ):
                {knowledge_base_text}

                ИНСТРУКЦИИ:
                1. Сначала проанализируй запрос и данные в блоке [РАССУЖДЕНИЯ]. Найди все подходящие произведения.
                2. В блоке [ОТВЕТ] сформируй итоговый результат.
                3. **СТРОГОЕ ПРАВИЛО:** Если найдены произведения, ты ОБЯЗАН разделить их на две группы:
                   - 🎬 **Фильмы** (если в поле 'Тип' указано Фильм)
                   - 🦄 **Мультфильмы** (если в поле 'Тип' указано Мультфильм или Анимация)
                4. Для каждого произведения создай красивую HTML карточку. Не используй Markdown таблицы. Используй `<div>` с inline-стилями.
                5. Если данных нет, ответь: "К сожалению, эта информация не найдена в архиве."

                ПРИМЕР ФОРМАТА ОТВЕТА:
                [РАССУЖДЕНИЯ]
                Пользователь ищет... В базе найдено...
                [ОТВЕТ]
                <h3>🎬 Фильмы</h3>
                <div class="answer-card">
                    <b>Название фильма</b> <span class="movie-tag">Фильм</span><br>
                    <i>Год: 2000 | Жанр: Фэнтези</i><br>
                    Описание...
                </div>
                
                <h3>🦄 Мультфильмы</h3>
                <div class="answer-card">
                    <b>Название мультфильма</b> <span class="cartoon-tag">Мультфильм</span><br>
                    ...
                </div>
                """
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": f"{prompt}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}"}],
                    temperature=0.1,
                    max_tokens=3000
                )
                
                answer = response.choices[0].message.content
                
                try:
                    parts = answer.split("[ОТВЕТ]")
                    reasoning_text = parts[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer_html = parts[1].strip()
                    
                    st.markdown(f'<div class="reasoning-box"><b>🕵️ Анализ запроса:</b><br>{reasoning_text}</div>', unsafe_allow_html=True)
                    st.markdown(final_answer_html, unsafe_allow_html=True)
                    
                except ValueError:
                    st.markdown(answer, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Ошибка: {e}</div>', unsafe_allow_html=True)

    elif not user_query and ask_button:
        st.warning("Пожалуйста, введите вопрос!")
