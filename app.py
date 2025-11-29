import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Пиксель", page_icon="✨", layout="wide")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
body, .stApp { font-family: 'Nunito', sans-serif !important; color: #333; }

h1 { color: #0e1117; font-size: 2rem !important; }
h3 { color: #1f77b4; font-size: 1.4rem !important; margin-top: 25px; border-bottom: 2px solid #eee; padding-bottom: 10px; }

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

.answer-card {
    background-color: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.answer-card:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.1); transition: 0.3s; }

.reasoning-box {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 15px;
    margin-bottom: 20px;
    font-size: 0.9rem;
    color: #856404;
    border-radius: 4px;
    white-space: pre-wrap;
}

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
        # Убедитесь, что файл ПроизведенияП.csv существует
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            knowledge_base += f"Название: {work.get('Name', 'Без названия')}\n"
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

st.markdown("<h1>✨ Умный ассистент Пиксель</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input(
        label=" ",
        placeholder="Например: Какая продолжительность у Пиратов Карибского моря?",
        key="user_input_box",
        label_visibility="collapsed"
    )
with col2:
    ask_button = st.button("Найти", use_container_width=True)

knowledge_base_text = create_knowledge_base()

if knowledge_base_text and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        # ИСПОЛЬЗУЕМ АКТУАЛЬНУЮ МОДЕЛЬ GROQ
        model_name = "llama-3.3-70b-versatile" 
    except Exception as e:
        st.error(f"Ошибка клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Пиксель ищет ответ..."):
            try:
                # --- ОБНОВЛЕННЫЙ ПРОМПТ ---
                prompt = f"""
                Ты - умный поисковый ассистент по базе данных фильмов.
                
                ТВОЯ ЦЕЛЬ: Найти информацию в предоставленном тексте и вернуть её строго в заданном формате.

                ДАННЫЕ:
                {knowledge_base_text}

                ИНСТРУКЦИЯ:
                1. Если запрос требует ФИЛЬТРАЦИИ (например, "выше 8.0", "после 2010 года"):
                   - Сравнивай числа строго ( > 8.0 это 8.1, а не 8.0).
                   - В [РАССУЖДЕНИЯ] пиши логику проверки (Пример: "Рейтинг 7.1 < 8.0 -> ОТКЛОНЕН").
                
                2. Если запрос - КОНКРЕТНЫЙ ВОПРОС (например, "Какая продолжительность у фильма Х?", "Кто режиссер Y?"):
                   - Найди этот фильм.
                   - В [РАССУЖДЕНИЯ] напиши просто: "Найден фильм Х, извлекаю запрашиваемую информацию".
                   - В [ОТВЕТ] верни карточку этого фильма.

                ВАЖНО: ВСЕГДА соблюдай структуру ответа, даже если нашел всего 1 фильм.

                СТРУКТУРА ОТВЕТА (СТРОГО):
                [РАССУЖДЕНИЯ]
                Текст рассуждений здесь.

                [ОТВЕТ]
                <div class="answer-card">
                    <b>Название фильма</b><br>
                    <i>Год: ... | Рейтинг: ... | Бюджет: ...</i><br>
                    Продолжительность: ... (или другая важная инфо)<br>
                    ...
                </div>
                (Повтори div для каждого найденного фильма)

                [ИТОГ]
                Список названий через запятую.
                """

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Запрос пользователя: {user_query}"}
                    ],
                    temperature=0.1, # Низкая температура для точности
                    max_tokens=2000
                )
                
                full_text = response.choices[0].message.content
                
                # --- УЛУЧШЕННЫЙ ПАРСИНГ ---
                reasoning = "Логика не предоставлена"
                answer_html = ""
                summary = ""

                # Пытаемся разбить текст более безопасно
                if "[РАССУЖДЕНИЯ]" in full_text and "[ОТВЕТ]" in full_text:
                    parts = full_text.split("[ОТВЕТ]")
                    reasoning_part = parts[0].split("[РАССУЖДЕНИЯ]")
                    if len(reasoning_part) > 1:
                        reasoning = reasoning_part[1].strip()
                    
                    rest = parts[1]
                    if "[ИТОГ]" in rest:
                        subparts = rest.split("[ИТОГ]")
                        answer_html = subparts[0].strip()
                        summary = subparts[1].strip()
                    else:
                        answer_html = rest.strip()
                else:
                    # Если модель сбилась и не выдала теги, показываем всё как есть
                    answer_html = full_text.replace("[РАССУЖДЕНИЯ]", "").replace("[ИТОГ]", "")

                # Вывод
                with st.expander("🕵️ Посмотреть логику отбора (Рассуждения)", expanded=False):
                    st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)
                
                if answer_html:
                    st.markdown(answer_html, unsafe_allow_html=True)
                else:
                    st.warning("Ничего не найдено по вашему запросу.")
                
                if summary:
                    st.markdown(f'<div class="summary-box"><span class="summary-title">📑 Итоговый список:</span>{summary}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ошибка обработки: {e}")

elif not GROQ_API_KEY:
    st.warning("Требуется GROQ_API_KEY.")
