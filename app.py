import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

# --- CSS СТИЛИ (ДИЗАЙН) ---
css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
body, .stApp { font-family: 'Nunito', sans-serif !important; color: #333; }

/* Заголовки */
h1 { color: #0e1117; font-size: 2rem !important; }
h3 { color: #1f77b4; font-size: 1.4rem !important; margin-top: 25px; border-bottom: 2px solid #eee; padding-bottom: 10px; }

/* Поля ввода и кнопки */
.stTextInput input { border-radius: 12px; border: 1px solid #ddd; padding: 12px; }
.stButton button { 
    border-radius: 12px; 
    background-color: #007bff; 
    color: white; 
    font-weight: bold; 
    border: none;
    height: 49px; 
}
.stButton button:hover { background-color: #0056b3; }

/* Карточки ответов */
.answer-card {
    background-color: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.answer-card:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.1); transition: 0.3s; }

/* Блок рассуждений */
.reasoning-box {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 15px;
    margin-bottom: 20px;
    font-size: 0.9rem;
    color: #856404;
    border-radius: 4px;
}

/* Блок итогового списка */
.summary-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 20px;
    border-radius: 15px;
    margin-top: 30px;
}
.summary-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 10px; display: block; }
"""
st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def create_knowledge_base():
    try:
        # Читаем CSV
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            # Собираем данные в текстовый блок для промпта
            knowledge_base += f"Название: {work['Name']}\n"
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n"
            knowledge_base += f"Год: {work.get('Год выпуска', '0')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', '0')}\n"
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            knowledge_base += f"Бюджет: {work.get('Бюджет и сборы','не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Описание: {work.get('Описание', 'не указано')}\n"
        return knowledge_base
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None

# --- UI ---
st.markdown("<h1>✨ Умный ассистент Пиксель</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input(
        label=" ",
        placeholder="Например: Фильмы после 2015 года с рейтингом ниже 7...",
        key="user_input_box",
        label_visibility="collapsed"
    )
with col2:
    ask_button = st.button("Найти", use_container_width=True)

knowledge_base_text = create_knowledge_base()

# --- ЛОГИКА ЗАПРОСА ---
if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "llama-3.3-70b-versatile" # Используем мощную модель
    except Exception as e:
        st.error(f"Ошибка клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Пиксель выполняет строгую фильтрацию..."):
            try:
                # --- СТРОГИЙ ПРОМПТ ---
                prompt = f"""
                Ты - строгий SQL-аналитик данных Disney.
                
                ТВОЯ ЦЕЛЬ: Отфильтровать список "ДАННЫЕ" согласно условиям пользователя и вывести результат.
                
                АЛГОРИТМ РАБОТЫ (СТРОГО):
                1. Прочитай запрос пользователя. Выдели условия фильтрации.
                   - Если сказано "после 2015", значит Год > 2015 (2015 НЕ ВКЛЮЧАТЬ).
                   - Если сказано "ниже 7.0", значит Рейтинг < 7.0 (7.0, 7.1 и 7.3 НЕ ВКЛЮЧАТЬ).
                   - Если сказано "Фильмы", значит Тип должен быть строго "Фильм" (Игнорируй "Мультфильм", "Анимация").
                
                2. Пройди по каждому элементу в базе данных и провериь условия.
                   - Пример проверки: "Круэлла": Рейтинг 7.3. Условие < 7.0. Результат: ОТКАЗАТЬ.
                   - Пример проверки: "Дамбо": Рейтинг 6.3. Условие < 7.0. Результат: ПРИНЯТЬ.
                
                3. Формат вывода:
                   В блоке [РАССУЖДЕНИЯ]: Перечисли, какие фильмы ты проверил и почему отклонил (например: "Круэлла (7.3 >= 7.0) -> Отклонен").
                   В блоке [ОТВЕТ]: Выведи ТОЛЬКО те карточки, которые прошли проверку.
                   В блоке [ИТОГ]: Список названий прошедших проверку.

                ОФОРМЛЕНИЕ [ОТВЕТ]:
                Используй HTML:
                <div class="answer-card">
                    <b>Название</b><br>
                    <i>Год: ... | Рейтинг: ...</i><br>
                    ...
                </div>
                
                Если пользователь просил только Фильмы, НЕ создавай заголовок "Мультфильмы".
                Если ничего не найдено после строгой фильтрации, напиши "Ничего не найдено по заданным критериям."

                ДАННЫЕ:
                {knowledge_base_text}
                """

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Запрос: {user_query}"}
                    ],
                    temperature=0.0, # НОЛЬ - чтобы модель не фантазировала
                    max_tokens=3000
                )
                
                full_text = response.choices[0].message.content
                
                # --- РАЗБОР ОТВЕТА ---
                reasoning = ""
                answer_html = ""
                summary = ""

                # Пытаемся разбить текст по тегам
                if "[РАССУЖДЕНИЯ]" in full_text:
                    parts = full_text.split("[ОТВЕТ]")
                    reasoning = parts[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    if len(parts) > 1:
                        rest = parts[1]
                        if "[ИТОГ]" in rest:
                            subparts = rest.split("[ИТОГ]")
                            answer_html = subparts[0].strip()
                            summary = subparts[1].strip()
                        else:
                            answer_html = rest.strip()
                else:
                    # Если модель забыла теги, выводим как есть
                    answer_html = full_text

                # --- ВЫВОД НА ЭКРАН ---
                with st.expander("🕵️ Посмотреть логику отбора (Рассуждения)", expanded=False):
                    st.markdown(f'<div class="reasoning-box">{reasoning.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                
                if answer_html:
                    st.markdown(answer_html, unsafe_allow_html=True)
                
                if summary:
                    st.markdown(f'<div class="summary-box"><span class="summary-title">📑 Итоговый список:</span>{summary.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ошибка обработки: {e}")

elif not GROQ_API_KEY:
    st.warning("Требуется GROQ_API_KEY.")
