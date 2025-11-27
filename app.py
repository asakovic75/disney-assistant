import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

def local_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Критическая ошибка: не найден файл стилей '{file_name}'!")

local_css("style.css")

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
    placeholder="Спросите что-нибудь о произведениях Disney...",
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
        with st.spinner("Ищу ответ..."):
            try:
                prompt = f"""Твоя роль - быть сверх-точным ассистентом-базой данных по фильмам Disney.

СТРОГИЕ ИНСТРУКЦИИ:
1.  **НИКАКИХ ДОГАДОК:** Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленных ниже "Данных".
2.  **ЕСЛИ ДАННЫХ НЕТ:** Если в данных нет ответа, твой ЕДИНСТВЕННЫЙ ответ должен быть: "К сожалению, эта информация не найдена в архиве."
3.  **РАЗЛИЧАЙ ТИПЫ (САМОЕ ВАЖНОЕ ПРАВИЛО):** Если пользователь спрашивает про "фильм", ты ОБЯЗАН искать только среди записей с `Тип: Фильм` и ПОЛНОСТЬЮ ИГНОРИРОВАТЬ все мультфильмы. И наоборот. В твоих рассуждениях должны быть ТОЛЬКО релевантные типы.
4.  **ФОРМАТ ОТВЕТА (ОБЯЗАТЕЛЬНО):** Твой ответ ДОЛЖЕН состоять из двух блоков: `[РАССУЖДЕНИЯ]` и `[ОТВЕТ]`.
    -   **В блоке `[РАССУЖДЕНИЯ]`:** Сначала **процитируй все релевантные (уже отфильтрованные!) записи из 'Данных'**. Затем, на основе этих цитат, сделай краткий логический вывод.
    -   **В блоке `[ОТВЕТ]`:** Сформулируй финальный, чистый ответ. **Если ответ представляет собой список, отформатируй его вертикально, используя дефис (-) и перенос строки для каждого элемента.**

ПРИМЕР №1 (простой ответ):
[РАССУЖДЕНИЯ]
Вопрос касается бюджета 'Короля Льва'. Я нашел следующую запись:
-----
Название: Король лев
Бюджет и сборы: $45 млн / $968 млн
Из этой записи видно, что бюджет составляет $45 млн.
[ОТВЕТ]
Бюджет мультфильма "Король лев" составляет $45 млн.

ПРИМЕР №2 (ответ списком):
[РАССУЖДЕНИЯ]
Вопрос о фильмах 2019 года. Я нашел следующие записи с `Тип: Фильм` и `Год выпуска: 2019`:
-----
Название: Король Лев: Новая глава
Год выпуска: 2019
Тип: Фильм
-----
Название: Аладдин: Новое желание
Год выпуска: 2019
Тип: Фильм
Я нашел два фильма, соответствующих критериям.
[ОТВЕТ]
Фильмы, выпущенные в 2019 году:
- Король Лев: Новая глава
- Аладдин: Новое желание

ДАННЫЕ:
{knowledge_base_text}

ВОПРОС: {user_query}

ОТВЕТ В СТРОГОМ ФОРМАТЕ:"""

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                answer = response.choices[0].message.content

                try:
                    reasoning_part, final_answer_part = answer.split("[ОТВЕТ]")
                    reasoning_text = reasoning_part.replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer_text = final_answer_part.strip()
                    
                    reasoning_html = reasoning_text.replace('\n', '<br>')
                    final_answer_html = final_answer_text.replace('\n', '<br>')

                    full_response_html = f"{reasoning_html}<br><br><hr><br><strong>{final_answer_html}</strong>"
                except ValueError:
                    full_response_html = answer.replace("[РАССУЖДЕНИЯ]", "").replace("[ОТВЕТ]", "").replace('\n', '<br>').strip()

                answer_placeholder.markdown(f'<div class="answer-text">{full_response_html}</div>', unsafe_allow_html=True)

            except Exception as e:
                answer_placeholder.markdown(f'<div class="error-message">❌ Ошибка: {e}</div>', unsafe_allow_html=True)
    elif not user_query and ask_button:
        answer_placeholder.markdown('<div class="warning-message">⚠️ Введите вопрос!</div>', unsafe_allow_html=True)
