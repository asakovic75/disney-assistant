import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

body, .stApp {
    font-family: 'Nunito', sans-serif !important;
    background-color: #F8FAFC;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #1E293B;
}

[data-testid="stTextInput"] {
    background: #FFFFFF !important;
    border-radius: 15px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    padding: 2px;
}

[data-testid="stTextInput"] input {
    color: #0F172A !important;
    font-size: 1rem !important;
}

.stButton button {
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 1rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.4);
}

.search-results-container {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #F1F5F9;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.movie-card {
    border-left: 4px solid #3B82F6;
    background: #F8FAFC;
    padding: 16px;
    margin-bottom: 16px;
    border-radius: 0 12px 12px 0;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E3A8A;
    margin-bottom: 8px;
    display: block;
}

.final-answer-box {
    background: linear-gradient(to right, #EFF6FF, #DBEAFE);
    border: 1px solid #BFDBFE;
    border-radius: 16px;
    padding: 24px;
    color: #1E3A8A;
    font-size: 1.05rem;
    line-height: 1.6;
    font-weight: 500;
}

.section-header {
    color: #64748B;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    font-weight: 700;
}
"""
st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def create_knowledge_base():
    try:
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        
        # Считаем статистику программно, чтобы не зависеть от ИИ
        total_count = len(works_df)
        movies_count = len(works_df[works_df['Тип'].str.contains("Фильм", case=False, na=False)])
        cartoons_count = len(works_df[works_df['Тип'].str.contains("Мультфильм", case=False, na=False)])
        
        stats = {
            "total": total_count,
            "movies": movies_count,
            "cartoons": cartoons_count
        }

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
            
        return knowledge_base, stats
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None, None

st.markdown("### ✨ Умный ассистент Пиксель")

user_query = st.text_input(
    label=" ",
    placeholder="Например: Сколько всего фильмов в базе? или Фильмы с рейтингом ниже 7.0...",
    key="user_input_box",
    label_visibility="collapsed"
)

ask_button = st.button("Найти ответ", use_container_width=True)

knowledge_base_text, db_stats = create_knowledge_base()
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
                # Вставляем точную статистику прямо в промпт
                stats_info = f"""
                ТОЧНАЯ СТАТИСТИКА БАЗЫ (ДЛЯ ВОПРОСОВ О КОЛИЧЕСТВЕ):
                - Всего произведений: {db_stats['total']}
                - Фильмов (Тип: Фильм): {db_stats['movies']}
                - Мультфильмов (Тип: Мультфильм): {db_stats['cartoons']}
                Если спрашивают 'сколько всего', бери числа ОТСЮДА, не считай вручную.
                """

                prompt = f"""Ты - Пиксель, умный ассистент.

{stats_info}

СТРОГИЕ ПРАВИЛА ФИЛЬТРАЦИИ:
1. ТИП:
   - "Фильм" -> искать строго `Тип: Фильм`.
   - "Мультфильм" -> искать строго `Тип: Мультфильм`.
   - Если тип не указан -> искать везде.

2. ЧИСЛА (МАТЕМАТИКА):
   - "Рейтинг ниже 7.0" -> 7.3 ЗАПРЕЩЕНО. 6.9 РАЗРЕШЕНО.
   - "После 2015 года" -> 2015 ЗАПРЕЩЕНО. 2016 РАЗРЕШЕНО.

ФОРМАТ ВЫВОДА:
[РАССУЖДЕНИЯ]
ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:

🎬 [Название]
🏷️ Тип: [Тип]
🎭 Жанр: [жанр]
📅 Год выпуска: [год]
💰 Бюджет и сборы: [бюджет]
🔞 Рейтинг: [рейтинг]
⏱️ Продолжительность: [время]
🏢 Студия: [студия]
🏆 Награды: [награды]
👥 Персонажи: [персонажи]
🎵 Песни: [песни]
🎡 Диснейленд: [связь с парком]

(Выводи только подходящие записи. Если вопрос про количество - не выводи карточки, переходи к анализу)

АНАЛИЗ: [кратко]

[ОТВЕТ]
[Здесь только итоговый текст. Без **.]

ДАННЫЕ:
{knowledge_base_text}

ВОПРОС: {user_query}

ОТВЕТ:"""

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
                        parts = answer.split("ОТВЕТ:")
                        if len(parts) > 1:
                            reasoning_part = parts[0]
                            final_answer_part = parts[1]
                        else:
                            reasoning_part = answer
                            final_answer_part = "Смотрите результаты выше."

                    reasoning_text = reasoning_part.replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer_text = final_answer_part.replace("**", "").replace("*", "").strip()

                    reasoning_html = reasoning_text.replace('\n', '<br>')
                    reasoning_html = reasoning_html.replace('ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:', '')
                    
                    # Обработка карточек для красивого вывода
                    if '🎬' in reasoning_html:
                        reasoning_html = reasoning_html.replace('🎬', '</div><div class="movie-card"><span class="card-title">🎬')
                        if reasoning_html.startswith('</div>'):
                            reasoning_html = reasoning_html[6:]
                        reasoning_html += '</div>'
                        search_display = f"<div class='search-results-container'><div class='section-header'>🔍 АНАЛИЗ БАЗЫ ДАННЫХ</div>{reasoning_html}</div>"
                    else:
                        # Если карточек нет (например, вопрос про количество), не показываем пустой блок
                        search_display = ""

                    final_answer_html = final_answer_text.replace('\n', '<br>')

                    full_response_html = f"""
                    {search_display}
                    <div class='final-answer-box'>
                        <div class='section-header' style='color: #1E3A8A;'>🤖 ОТВЕТ ПИКСЕЛЯ</div>
                        {final_answer_html}
                    </div>
                    """
                except Exception:
                    full_response_html = f"<div class='final-answer-box'>{answer}</div>"

                answer_placeholder.markdown(full_response_html, unsafe_allow_html=True)

            except Exception as e:
                answer_placeholder.error(f"Произошла ошибка: {e}")

    elif not user_query and ask_button:
        answer_placeholder.warning("Пожалуйста, введите вопрос!")
