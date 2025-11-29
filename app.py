import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

# --- ДИЗАЙН И СТИЛИ (CSS) ---
css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
body, .stApp { font-family: 'Nunito', sans-serif !important; color: #333; }
h1 { color: #0e1117; font-size: 2rem !important; }
h3 { color: #1f77b4; font-size: 1.3rem !important; margin-top: 20px; }

/* Стили полей ввода и кнопок */
.stTextInput input { border-radius: 10px; border: 1px solid #ddd; padding: 10px; }
.stButton button { 
    border-radius: 10px; 
    background-color: #007bff; 
    color: white; 
    font-weight: bold; 
    border: none;
    height: 46px; 
}
.stButton button:hover { background-color: #0056b3; }

/* Стили для карточек ответов */
.answer-card {
    background-color: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: transform 0.2s;
}
.answer-card:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.1); }

/* Теги типов */
.tag { display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: bold; margin-bottom: 5px; }
.tag-film { background-color: #e3f2fd; color: #1565c0; }
.tag-cartoon { background-color: #fce4ec; color: #c2185b; }

/* Блок рассуждений */
.reasoning-box {
    background-color: #f8f9fa;
    border-left: 4px solid #ffc107;
    padding: 15px;
    margin-bottom: 25px;
    font-size: 0.9rem;
    color: #6c757d;
    border-radius: 0 8px 8px 0;
}
"""
st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def create_knowledge_base():
    try:
        # Проверьте правильность имени файла, у вас было 'ПроизведенияП.csv'
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            knowledge_base += f"Название: {work['Name']}\n"
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n"
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Год: {work.get('Год выпуска','не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', 'не указано')}\n"
            knowledge_base += f"Бюджет: {work.get('Бюджет и сборы','не указано')}\n"
            knowledge_base += f"Исполнители: {work.get('Исполнители','не указано')}\n"
            knowledge_base += f"Описание: {work.get('Описание', 'не указано')}\n"
        return knowledge_base
    except Exception as e:
        st.error(f"Ошибка при загрузке CSV: {e}")
        return None

# --- ИНТЕРФЕЙС ---
st.markdown("<h1>✨ Умный ассистент Пиксель</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input(label=" ", placeholder="Спросите о фильмах или мультфильмах Disney...", label_visibility="collapsed")
with col2:
    ask_button = st.button("Найти", use_container_width=True)

knowledge_base_text = create_knowledge_base()
answer_placeholder = st.empty()

# --- ЛОГИКА ---
if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        # АКТУАЛЬНАЯ МОДЕЛЬ (Llama 3.3)
        model_name = "llama-3.3-70b-versatile"
    except Exception as e:
        st.error(f"Ошибка клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Пиксель анализирует базу данных..."):
            try:
                # СТРОГИЙ ПРОМПТ
                prompt = f"""
                Ты - ассистент базы данных Disney.
                
                ТВОЯ ЗАДАЧА:
                1. Проанализируй запрос пользователя и предоставленные "ДАННЫЕ".
                2. Сформируй ответ в два этапа: сначала [РАССУЖДЕНИЯ], потом [ОТВЕТ].
                
                СТРОГИЕ ПРАВИЛА ДЛЯ [ОТВЕТ]:
                1. **ИСТОЧНИК:** Используй ТОЛЬКО предоставленные ниже ДАННЫЕ. Никаких догадок.
                2. **СТРУКТУРА:** Если найдены произведения, ты ОБЯЗАН разделить их на заголовки: "🎬 Фильмы" и "🦄 Мультфильмы" (смотри поле 'Тип').
                3. **ДИЗАЙН:** Для каждого произведения создай HTML-карточку с классом `answer-card`. Внутри используй теги <b>, <i>, <br>.
                4. Если данных нет, напиши: "К сожалению, эта информация не найдена в архиве."

                ПРИМЕР HTML ВЫВОДА (внутри [ОТВЕТ]):
                <h3>🎬 Фильмы</h3>
                <div class="answer-card">
                    <span class="tag tag-film">Фильм</span> <b>Название</b><br>
                    <i>Год: 2021 | Жанр: Фэнтези</i><br><br>
                    Описание...
                </div>

                ДАННЫЕ:
                {knowledge_base_text}
                """

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Вопрос: {user_query}"}
                    ],
                    temperature=0.1,
                    max_tokens=2500
                )
                
                full_response = response.choices[0].message.content
                
                # РАЗБОР ОТВЕТА
                try:
                    parts = full_response.split("[ОТВЕТ]")
                    reasoning = parts[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_html = parts[1].strip()
                except:
                    reasoning = ""
                    final_html = full_response.replace("[РАССУЖДЕНИЯ]", "").replace("[ОТВЕТ]", "")

                # ВЫВОД НА ЭКРАН
                if reasoning:
                    st.markdown(f'<div class="reasoning-box"><b>🔍 Анализ запроса:</b><br>{reasoning}</div>', unsafe_allow_html=True)
                
                st.markdown(final_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ошибка генерации: {e}")

elif not GROQ_API_KEY:
    st.warning("Не найден GROQ_API_KEY.")
elif not knowledge_base_text:
    st.error("База данных пуста или файл не найден.")
