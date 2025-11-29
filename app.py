import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');

body, .stApp {
    font-family: 'Nunito', sans-serif !important;
    background: transparent;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1 { font-size: 1.5rem !important; text-align: left; }
h3, h5 { font-size: 1.2rem !important; text-align: left; }

[data-testid="stTextInput"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

[data-testid="stTextInput"] input {
    background: transparent !important;
    color: #111111 !important;
    font-size: 0.9rem !important;
    padding: 10px 15px !important;
    border: none !important;
    outline: none !important;
}

.stButton button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-size: 0.9rem !important;
    font-weight: 700;
    background: #3B82F6 !important;
    color: white !important;
    border: none !important;
}

.reasoning-section {
    background-color: #F3F4F6;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
    font-size: 0.95rem;
    border-left: 5px solid #3B82F6;
    color: #374151;
}

.films-list {
    margin-top: 10px;
}

.final-answer-section {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 20px;
    color: #1E3A8A;
    font-weight: 500;
}

.error-message {
    background-color: #FEF2F2;
    color: #EF4444 !important;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}

.warning-message {
    background-color: #FFFBEB;
    color: #F59E0B !important;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
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
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы', 'не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска', 'не указано')}\n"
            knowledge_base += f"Диснейленд: {work.get('Диснейленд', 'не указано')}\n" 
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Исполнители: {work.get('Исполнители', 'не указано')}\n"
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

st.markdown("##### ✨ Умный ассистент Пиксель")

user_query = st.text_input(
    label=" ",
    placeholder="Спросите о фильмах или мультфильмах Disney...",
    key="user_input_box",
    label_visibility="collapsed"
)

ask_button = st.button("Найти", use_container_width=True, key="find_answer")

knowledge_base_text = create_knowledge_base()
answer_placeholder = st.empty()

if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct" 
    except Exception as e:
        st.error(f"Ошибка инициализации клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("✨ Пиксель ищет информацию..."):
            try:
                prompt = f"""Твоя роль - быть умным ассистентом Пикселем.

ПРАВИЛА ЛОГИКИ (ОЧЕНЬ СТРОГО):
1.  **МАТЕМАТИКА:**
    - "Рейтинг ниже 7.0": это значит СТРОГО МЕНЬШЕ (<) 7.0. Рейтинг 7.3 НЕ подходит. Рейтинг 7.0 НЕ подходит. Рейтинг 6.8 подходит.
    - "После 2015 года": это значит > 2015 (2016, 2017...).
2.  **ФИЛЬТР ТИПА:**
    - Если вопрос про "фильмы", игнорируй "Тип: Мультфильм".
    - Если вопрос про "мультфильмы", игнорируй "Тип: Фильм".
3.  **ЧИСТОТА ВЫВОДА:**
    - В блоке [РАССУЖДЕНИЯ] показывай ТОЛЬКО те карточки, которые прошли проверку математикой. Если фильм не подходит, НЕ показывай его.
    - Разделяй Рейтинг и Возраст на отдельные строки.

ФОРМАТ ОТВЕТА:
[РАССУЖДЕНИЯ]
🎬 [Название]
🏷️ Тип: [Фильм/Мультфильм]
🎭 Жанр: [Жанр]
🔞 Возраст: [Возраст]
⭐ Рейтинг: [Рейтинг]
📅 Год выпуска: [Год]
⏱️ Продолжительность: [Время]
💰 Бюджет и сборы: [Бюджет]
🏆 Награды: [Награды]
👥 Персонажи: [Персонажи]
🎥 Исполнители: [Исполнители]
🎡 Диснейленд: [Парк]
🏢 Студия: [Студия]
🎵 Песни: [Песни]

(Повтори блок для каждого подходящего фильма)

АНАЛИЗ: [Краткий итог]

[ОТВЕТ]
[Здесь напиши итоговый список. НЕ используй символы ** или []. Просто текст.]

ДАННЫЕ:
{knowledge_base_text}

ВОПРОС: {user_query}
"""

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2500
                )
                answer = response.choices[0].message.content

                try:
                    if "[ОТВЕТ]" in answer:
                        reasoning_part, final_answer_part = answer.split("[ОТВЕТ]")
                    else:
                        reasoning_part = answer
                        final_answer_part = "Смотрите детали выше."

                    reasoning_text = reasoning_part.replace("[РАССУЖДЕНИЯ]", "").strip()
                    reasoning_text = reasoning_text.replace("ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:", "").strip()
                    reasoning_text = reasoning_text.replace("[ПОИСКОВЫЕ РЕЗУЛЬТАТЫ]", "").strip()
                    
                    final_answer_text = final_answer_part.strip().replace("**", "").replace("[", "").replace("]", "")

                    reasoning_html = reasoning_text.replace('\n', '<br>')
                    final_answer_html = final_answer_text.replace('\n', '<br>')
                    
                    reasoning_html = reasoning_html.replace('🎬', '<br><span style="font-size: 1.3em;">🎬</span>')

                    full_response_html = f"""
                    <div class='reasoning-section'>
                        <h4 style='margin-top:0; color:#4B5563;'>🔍 Найденные карточки:</h4>
                        <div class='films-list'>
                            {reasoning_html}
                        </div>
                    </div>
                    <div class='final-answer-section'>
                        <h4 style='margin-top:0;'>🤖 Ответ Пикселя:</h4>
                        <b>{final_answer_html}</b>
                    </div>
                    """
                except ValueError:
                    full_response_html = answer.replace("\n", "<br>")

                answer_placeholder.markdown(full_response_html, unsafe_allow_html=True)

            except Exception as e:
                answer_placeholder.markdown(f'<div class="error-message">❌ Ошибка: {e}</div>', unsafe_allow_html=True)
                
    elif not user_query and ask_button:
        answer_placeholder.markdown('<div class="warning-message">⚠️ Пожалуйста, введите вопрос!</div>', unsafe_allow_html=True)
