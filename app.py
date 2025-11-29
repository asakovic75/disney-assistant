import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# --- Настройки ---
st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- CSS (минималистичный, как в оригинале, но рабочий) ---
st.markdown("""
<style>
.answer-text {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    color: #000000;
}
.warning-message {
    color: #ffbd45;
    font-weight: bold;
}
.error-message {
    color: #ff4b4b;
    font-weight: bold;
}
hr {
    margin: 20px 0;
    border: 0;
    border-top: 1px solid #ccc;
}
</style>
""", unsafe_allow_html=True)

# --- Загрузка данных ---
@st.cache_data
def create_knowledge_base():
    try:
        # Читаем файл
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        
        # 1. СЧИТАЕМ СТАТИСТИКУ ПРОГРАММНО (для точности)
        total = len(works_df)
        movies = len(works_df[works_df['Тип'].str.contains("Фильм", case=False, na=False)])
        cartoons = len(works_df[works_df['Тип'].str.contains("Мультфильм", case=False, na=False)])
        
        stats_text = f"Всего записей: {total}. Из них Фильмов: {movies}, Мультфильмов: {cartoons}."

        # Формируем текст базы
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            knowledge_base += f"Название: {work['Name']}\n"
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n" # Тип важен, ставим выше
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска', 'не указано')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', 'не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы', 'не указано')}\n"
            knowledge_base += f"Диснейленд: {work.get('Диснейленд', 'не указано')}\n"
            knowledge_base += f"Исполнители: {work.get('Исполнители', 'не указано')}\n"
            knowledge_base += f"Награды: {work.get('Награды', 'не указано')}\n"
            knowledge_base += f"Персонажи: {work.get('Персонажи', 'не указано')}\n"
            knowledge_base += f"Песни: {work.get('Песни', 'не указано')}\n"
            knowledge_base += f"Продолжительность: {work.get('Продолжительность', 'не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            
        return knowledge_base, stats_text
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None, None

# --- Интерфейс ---
st.markdown("##### ✨ Умный ассистент Пиксель")

user_query = st.text_input(
    label=" ",
    placeholder="Спросите что-нибудь о произведениях Disney...",
    key="user_input_box",
    label_visibility="collapsed"
)

ask_button = st.button("Найти", use_container_width=True, key="find_answer")

knowledge_base_text, db_stats = create_knowledge_base()
answer_placeholder = st.empty()

# --- Логика ---
if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    except Exception as e:
        st.error(f"Ошибка инициализации клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Ищу ответ..."):
            try:
                # ВОТ ТУТ ГЛАВНЫЕ ПРАВИЛА
                prompt = f"""Твоя роль - быть точным аналитиком базы данных Disney.

ТВОИ ДАННЫЕ О КОЛИЧЕСТВЕ (ИСПОЛЬЗУЙ ИХ ДЛЯ ОТВЕТОВ "СКОЛЬКО"):
{db_stats}

ИНСТРУКЦИИ ПО ПОИСКУ И ЛОГИКЕ:
1. **ФИЛЬТР ТИПА (СТРОГО):**
   - Если вопрос про "ФИЛЬМЫ" (кино) -> Ищи ТОЛЬКО где `Тип: Фильм`. Игнорируй `Тип: Мультфильм`.
   - Если вопрос про "МУЛЬТФИЛЬМЫ" -> Ищи ТОЛЬКО где `Тип: Мультфильм`.
   
2. **ЛОГИЧЕСКИЕ ОПЕРАЦИИ (МАТЕМАТИКА):**
   - "Рейтинг НИЖЕ 7.0": 7.3 > 7.0 (НЕТ), 6.9 < 7.0 (ДА).
   - "ПОСЛЕ 2015 года": 2015 (НЕТ), 2016 (ДА).
   - Сравнивай числа внимательно.

3. **ФОРМАТ ОТВЕТА:**
   Ты ОБЯЗАН использовать два блока:
   [РАССУЖДЕНИЯ]
   (Здесь перечисли найденные карточки или напиши, как ты считал. Если карточек много, покажи список. Используй смайлики)
   [ОТВЕТ]
   (Здесь финальный краткий ответ текстом. Без символов **)

ДАННЫЕ:
{knowledge_base_text}

ВОПРОС: {user_query}

ОТВЕТ:"""

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, # Ставим 0 для максимальной точности
                    max_tokens=2500
                )
                answer = response.choices[0].message.content

                try:
                    # Разделяем ответ на рассуждения и итог
                    if "[ОТВЕТ]" in answer:
                        reasoning_part, final_answer_part = answer.split("[ОТВЕТ]")
                    else:
                        # Если вдруг модель забыла тег
                        reasoning_part = answer
                        final_answer_part = "Смотрите выше"

                    reasoning_text = reasoning_part.replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer_text = final_answer_part.replace("**", "").strip() # Убираем жирный шрифт

                    # Формируем HTML
                    reasoning_html = reasoning_text.replace('\n', '<br>')
                    final_answer_html = final_answer_text.replace('\n', '<br>')

                    # Собираем все вместе
                    full_response_html = f"""
                    <div style="color: #555; font-size: 0.9em;"><b>🔍 Рассуждения и поиск:</b><br>{reasoning_html}</div>
                    <hr>
                    <div class="answer-text"><b>🤖 Ответ:</b><br>{final_answer_html}</div>
                    """
                except ValueError:
                    full_response_html = answer.replace("\n", "<br>")

                answer_placeholder.markdown(full_response_html, unsafe_allow_html=True)

            except Exception as e:
                answer_placeholder.markdown(f'<div class="error-message">❌ Ошибка: {e}</div>', unsafe_allow_html=True)
                
    elif not user_query and ask_button:
        answer_placeholder.markdown('<div class="warning-message">Введите вопрос!</div>', unsafe_allow_html=True)
