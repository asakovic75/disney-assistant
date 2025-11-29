import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

body, .stApp {
    font-family: 'Nunito', sans-serif !important;
    background: transparent;
}

[data-testid="stHeader"] {
    background: transparent;
}

h4 { margin-top: 0 !important; }

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
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 25px;
    font-size: 0.95rem;
    color: #374151;
}

.card {
    background: #FFFFFF;
    border-left: 4px solid #3B82F6;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-radius: 0 8px 8px 0;
}

.card-title {
    font-size: 1.2em;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
    display: block;
}

.final-answer-section {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 20px;
    color: #1E3A8A;
    font-size: 1.05rem;
    line-height: 1.6;
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
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n"
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска', 'не указано')}\n"
            knowledge_base += f"Продолжительность: {work.get('Продолжительность', 'не указано')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', 'не указано')}\n"
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы', 'не указано')}\n"
            knowledge_base += f"Награды: {work.get('Награды', 'не указано')}\n"
            knowledge_base += f"Персонажи: {work.get('Персонажи', 'не указано')}\n"
            knowledge_base += f"Исполнители: {work.get('Исполнители', 'не указано')}\n"
            knowledge_base += f"Диснейленд: {work.get('Диснейленд', 'не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            knowledge_base += f"Песни: {work.get('Песни', 'не указано')}\n"
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
                prompt = f"""Ты - умный ассистент Пиксель. Твоя цель - найти точную информацию в базе данных Disney.

ПРАВИЛА ЛОГИКИ (СТРОГО):
1.  **Фильтр "Тип":** 
    - Если спрашивают "фильм", ищи только `Тип: Фильм`.
    - Если спрашивают "мультфильм", ищи только `Тип: Мультфильм`.
2.  **Фильтр "Числа":**
    - "Рейтинг ниже 7.0" -> 7.0 включаем, 7.3 исключаем.
    - "После 2015 года" -> 2016, 2017... (2015 не включаем, если не сказано "с 2015").
3.  **Точность:** В блок [РАССУЖДЕНИЯ] включай ТОЛЬКО те карточки, которые на 100% соответствуют условиям. Если фильм "почти подходит", НЕ показывай его.

ФОРМАТ ОТВЕТА:

[РАССУЖДЕНИЯ]
ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:

🎬 Название: [Название]
🏷️ Тип: [Фильм/Мультфильм]
🎭 Жанр: [Жанр]
🔞 Возраст: [Возраст]
📅 Год выпуска: [Год]
⏱️ Продолжительность: [Время]
⭐ Рейтинг: [Рейтинг]
💰 Бюджет и сборы: [Деньги]
🏆 Награды: [Награды]
👥 Персонажи: [Персонажи]
🎥 Исполнители: [Актеры]
🎡 Диснейленд: [Парк]
🏢 Студия: [Студия]
🎵 Песни: [Песни]

(Повторить для каждого найденного результата)

АНАЛИЗ: [Краткий вывод]

[ОТВЕТ]
[Здесь напиши итоговый ответ. НЕ используй Markdown символы вроде ** или __. Если нужно выделить текст, просто пиши его. Если список - делай каждый пункт с новой строки.]

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
                        final_answer_part = "Подробности выше."

                    reasoning_text = reasoning_part.replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer_text = final_answer_part.strip()
                    
                    reasoning_html = reasoning_text.replace('\n', '<br>')
                    
                    reasoning_html = reasoning_html.replace('🎬 Название:', '</div><div class="card"><span class="card-title">🎬')
                    reasoning_html = reasoning_html.replace('ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:<br><br></div>', 'ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:') 
                    
                    if '<div class="card">' not in reasoning_html and '🎬' in reasoning_html:
                         reasoning_html = reasoning_html.replace('🎬', '<div class="card"><span class="card-title">🎬')

                    final_answer_html = final_answer_text.replace('\n', '<br>')
                    final_answer_html = final_answer_html.replace('**', '').replace('__', '') 
                    
                    full_response_html = f"""
                    <div class='reasoning-section'>
                        <h4 style='color:#4B5563;'>🔍 Результаты поиска:</h4>
                        {reasoning_html}
                        </div>
                    </div>
                    <div class='final-answer-section'>
                        <h4 style='color:#1E3A8A;'>🤖 Ответ:</h4>
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
