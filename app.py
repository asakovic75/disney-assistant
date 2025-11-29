import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# Настройка страницы
st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

# --- CSS СТИЛИ (ДИЗАЙН) ---
css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
body, .stApp { font-family: 'Nunito', sans-serif !important; color: #333; }

/* Заголовки */
h1 { color: #0e1117; font-size: 2rem !important; }
h3 { color: #1f77b4; font-size: 1.4rem !important; margin-top: 25px; border-bottom: 2px solid #eee; padding-bottom: 10px; }

/* Кнопки и ввод */
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

/* Теги */
.tag { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-bottom: 8px; }
.tag-film { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
.tag-cartoon { background-color: #fce4ec; color: #c2185b; border: 1px solid #f8bbd0; }

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

# --- ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ---
@st.cache_data
def create_knowledge_base():
    try:
        # Убедитесь, что файл называется именно так
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            knowledge_base += f"Название: {work['Name']}\n"
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы','не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска','не указано')}\n"
            knowledge_base += f"Диснейленд: {work.get('Диснейленд','не указано')}\n"
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

# --- ИНТЕРФЕЙС ---
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
    ask_button = st.button("Найти", use_container_width=True, key="find_answer")

knowledge_base_text = create_knowledge_base()
answer_placeholder = st.empty()

# --- ЛОГИКА ---
if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "llama-3.3-70b-versatile"
    except Exception as e:
        st.error(f"Ошибка инициализации клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Анализирую данные..."):
            try:
                # --- ПРОМПТ ---
                prompt = f"""
                Ты - строгий аналитик базы данных Disney.
                Твоя задача - отвечать ТОЛЬКО на основе предоставленных ДАННЫХ.
                
                ИНСТРУКЦИИ ПО ФИЛЬТРАЦИИ:
                1. Сравнивай числа (Год, Рейтинг) математически точно. 
                   - Если просят "после 2015", то 2014, 2015 - НЕ подходят. Подходят 2016, 2017...
                   - Если просят "рейтинг ниже 7", то 7.0, 7.2 - НЕ подходят. Подходят 6.9, 6.8...
                2. Если в данных нет поля, считай его "не указано".
                
                ФОРМАТ ВЫВОДА (Строго соблюдай разделители):
                
                [РАССУЖДЕНИЯ]
                (Здесь напиши ход мыслей: какие фильмы проверил, почему они подходят или не подходят по критериям)
                
                [ОТВЕТ]
                (Здесь HTML код для карточек. ОБЯЗАТЕЛЬНО разделяй на заголовки <h3>🎬 Фильмы</h3> и <h3>🦄 Мультфильмы</h3> на основе поля 'Тип'.
                Используй класс <div class="answer-card"> для оформления.)
                
                [ИТОГ]
                (Здесь напиши краткий маркированный список только названий найденных произведений в качестве резюме)
                
                ДАННЫЕ ДЛЯ АНАЛИЗА:
                {knowledge_base_text}
                """

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Запрос пользователя: {user_query}"}
                    ],
                    temperature=0.0, # Ноль для максимальной точности
                    max_tokens=3000
                )
                
                full_text = response.choices[0].message.content
                
                # --- ПАРСИНГ ОТВЕТА ---
                # 1. Извлекаем Рассуждения
                try:
                    parts_1 = full_text.split("[ОТВЕТ]")
                    reasoning = parts_1[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    rest_of_text = parts_1[1] if len(parts_1) > 1 else ""
                except:
                    reasoning = full_text
                    rest_of_text = ""

                # 2. Извлекаем Ответ (карточки) и Итог (список)
                try:
                    parts_2 = rest_of_text.split("[ИТОГ]")
                    cards_html = parts_2[0].strip()
                    summary_list = parts_2[1].strip() if len(parts_2) > 1 else ""
                except:
                    cards_html = rest_of_text
                    summary_list = ""

                # --- ВЫВОД НА ЭКРАН ---
                
                # 1. Рассуждения (можно скрыть в expander, но вы просили анализ)
                with st.expander("🕵️ Посмотреть анализ (Рассуждения ИИ)", expanded=False):
                    st.markdown(f'<div class="reasoning-box">{reasoning.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                
                # 2. Карточки
                if cards_html:
                    st.markdown(cards_html, unsafe_allow_html=True)
                else:
                    st.warning("К сожалению, по вашему запросу ничего не найдено.")

                # 3. Итоговый список
                if summary_list:
                    st.markdown(f"""
                    <div class="summary-box">
                        <span class="summary-title">📑 Итоговый список:</span>
                        {summary_list.replace(chr(10), "<br>")}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Ошибка: {e}</div>', unsafe_allow_html=True)

elif not GROQ_API_KEY:
    st.error("Пожалуйста, добавьте GROQ_API_KEY в переменные окружения.")
