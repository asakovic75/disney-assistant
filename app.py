import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# --- 1. Настройки страницы (Чистый белый стиль) ---
st.set_page_config(page_title="Пиксель", page_icon="✨", layout="centered")

# --- 2. CSS: Простой дизайн ---
st.markdown("""
<style>
    /* Глобальный фон и цвет текста */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Заголовок */
    h1 {
        font-family: sans-serif;
        color: #333;
        font-weight: 700;
    }

    /* Стиль кнопки */
    .stButton > button {
        background-color: #000000; /* Черная кнопка для контраста */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #333333;
        color: #ffffff;
    }

    /* Блок РАССУЖДЕНИЯ (Технический вид) */
    .reasoning-box {
        background-color: #f4f4f4; /* Светло-серый фон */
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        font-family: 'Consolas', 'Courier New', monospace; /* Моноширинный шрифт */
        font-size: 0.85em;
        color: #444;
        line-height: 1.4;
    }

    /* Блок ОТВЕТ (Чистый вид) */
    .answer-box {
        background-color: #ffffff;
        border-left: 4px solid #000; /* Черная линия слева */
        padding: 20px;
        font-family: sans-serif;
        font-size: 1.1em;
        color: #000;
        line-height: 1.6;
    }
    
    /* Подзаголовки блоков */
    .block-label {
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.8em;
        color: #666;
        margin-bottom: 8px;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Инициализация ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def create_knowledge_base():
    try:
        # Читаем CSV (убедись, что файл лежит рядом со скриптом)
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        return works_df
    except Exception as e:
        st.error(f"Ошибка чтения файла 'ПроизведенияП.csv': {e}")
        return None

# --- 4. Интерфейс ---
st.title("Пиксель")
st.caption("Умный поиск по базе Disney")

user_query = st.text_input("Ваш запрос:", placeholder="Введите вопрос...")
ask_button = st.button("Найти")

works_df = create_knowledge_base()
answer_placeholder = st.empty()

# --- 5. Логика ---
if works_df is not None and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "meta-llama/llama-3.3-70b-versatile" # Или другая доступная модель
    except Exception as e:
        st.error(f"Ошибка клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Анализ данных..."):
            try:
                # Собираем контекст из таблицы
                context = ""
                for _, work in works_df.iterrows():
                    context += "---\n"
                    context += f"Название: {work.get('Name')}\n"
                    context += f"Тип: {work.get('Тип')} (Важно!)\n"
                    context += f"Год: {work.get('Год выпуска')}\n"
                    context += f"Рейтинг: {work.get('Рейтинг')}\n"
                    context += f"Жанр: {work.get('Жанр')}\n"
                    context += f"Описание: {work.get('Описание')}\n"

                # Промпт
                prompt = f"""
                Твоя роль - поисковый алгоритм по базе данных.
                
                ИНСТРУКЦИИ:
                1. СТРОГО РАЗЛИЧАЙ: "Фильм" (живое кино) и "Мультфильм" (анимация).
                   - Если просят фильмы -> исключи мультфильмы.
                   - Если просят мультфильмы -> исключи фильмы.
                2. Не придумывай данные. Используй только контекст.

                ФОРМАТ ВЫВОДА (Строго два блока):
                
                [РАССУЖДЕНИЯ]
                Напиши технический отчет о поиске.
                - Без жирного шрифта (без звездочек).
                - Без эмодзи.
                - Просто текст: какие записи проверил, какие подошли по году/рейтингу/типу, какие отсеял.
                
                [ОТВЕТ]
                Финальный ответ для пользователя. Четкий список или текст.

                КОНТЕКСТ ДАННЫХ:
                {context}

                ЗАПРОС: {user_query}
                """

                # Запрос к нейросети
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                full_text = response.choices[0].message.content

                # Обработка ответа
                try:
                    parts = full_text.split("[ОТВЕТ]")
                    reasoning = parts[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    answer = parts[1].strip()
                    
                    # Убираем звездочки (markdown bold) из рассуждений, чтобы было чисто
                    reasoning_clean = reasoning.replace("**", "").replace("*", "")
                    
                    # Преобразуем переносы строк в HTML
                    reasoning_html = reasoning_clean.replace('\n', '<br>')
                    answer_html = answer.replace('\n', '<br>')
                    
                except:
                    # Если формат нарушен
                    reasoning_html = "Ошибка формата ответа."
                    answer_html = full_text

                # Вывод
                st.markdown(f"""
                <div class="reasoning-box">
                    <span class="block-label">Логика поиска:</span>
                    {reasoning_html}
                </div>
                
                <div class="answer-box">
                    <span class="block-label">Результат:</span>
                    {answer_html}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")

    elif not user_query and ask_button:
        st.warning("Введите запрос.")

elif not works_df:
    st.error("Файл 'ПроизведенияП.csv' не найден.")
elif not GROQ_API_KEY:
    st.error("API ключ не найден.")
