import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# 1. Настройка страницы
st.set_page_config(page_title="Пиксель — Disney Assistant", page_icon="✨", layout="wide")

# Получаем ключ из секретов или переменных окружения
# Рекомендуется использовать st.secrets["GROQ_API_KEY"] для деплоя
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# 2. CSS Стилизация
st.markdown("""
<style>
    /* Основной фон и текст */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Заголовок */
    h1, h2, h3, h4, h5 {
        color: #e0e0ff !important;
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 0 0 10px rgba(100, 200, 255, 0.5);
    }

    /* Поле ввода */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid #5d54a4;
        border-radius: 12px;
    }
    .stTextInput input:focus {
        border-color: #9d50bb;
        box-shadow: 0 0 10px #9d50bb;
    }

    /* Кнопка */
    .stButton button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(37, 117, 252, 0.4);
    }

    /* Блок ответа */
    .answer-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00d2ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-size: 1.1em;
        line-height: 1.6;
    }
    
    /* Сообщение об ошибке */
    .error-box {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid red;
        padding: 15px;
        border-radius: 10px;
        color: #ffcccc;
    }
</style>
""", unsafe_allow_html=True)

# 3. Загрузка базы знаний
@st.cache_data
def create_knowledge_base():
    file_path = "ПроизведенияП.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден! Пожалуйста, добавьте его в папку с проектом.")
        return None
        
    try:
        works_df = pd.read_csv(file_path).astype(str).fillna('не указано')
        knowledge_base = ""
        for _, work in works_df.iterrows():
            knowledge_base += "-----\n"
            knowledge_base += f"Название: {work.get('Name', 'не указано')}\n"
            knowledge_base += f"Бюджет и сборы: {work.get('Бюджет и сборы', 'не указано')}\n"
            knowledge_base += f"Возраст: {work.get('Возраст', 'не указано')}\n"
            knowledge_base += f"Год выпуска: {work.get('Год выпуска', 'не указано')}\n"
            knowledge_base += f"Жанр: {work.get('Жанр', 'не указано')}\n"
            knowledge_base += f"Награды: {work.get('Награды', 'не указано')}\n"
            knowledge_base += f"Персонажи: {work.get('Персонажи', 'не указано')}\n"
            knowledge_base += f"Песни: {work.get('Песни', 'не указано')}\n"
            knowledge_base += f"Продолжительность: {work.get('Продолжительность', 'не указано')}\n"
            knowledge_base += f"Рейтинг: {work.get('Рейтинг', 'не указано')}\n"
            knowledge_base += f"Студия: {work.get('Студия', 'не указано')}\n"
            # Важное поле для различения типа
            knowledge_base += f"Тип: {work.get('Тип', 'не указано')}\n" 
        return knowledge_base
    except Exception as e:
        st.error(f"❌ Ошибка при чтении CSV: {e}")
        return None

# Инициализация интерфейса
st.markdown("##### ✨ Умный ассистент Пиксель")
st.markdown("Задайте вопрос о вселенной Disney, и я проанализирую данные.")

# Поля ввода
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        label="Ваш вопрос",
        placeholder="Например: В каких мультфильмах есть Микки Маус?",
        key="user_input_box",
        label_visibility="collapsed"
    )
with col2:
    ask_button = st.button("Найти", use_container_width=True, key="find_answer")

# Загрузка базы
knowledge_base_text = create_knowledge_base()
answer_placeholder = st.empty()

# 4. Логика обработки запроса
if ask_button:
    if not GROQ_API_KEY:
        st.warning("⚠️ Не найден ключ API. Установите GROQ_API_KEY.")
    elif not user_query:
        st.warning("⚠️ Пожалуйста, введите вопрос!")
    elif not knowledge_base_text:
        st.error("⚠️ База знаний пуста.")
    else:
        try:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY
            )
            # Используем актуальную модель (Llama 3 70b или Mixtral обычно лучше справляются с русским)
            model_name = "llama3-70b-8192" 

            with st.spinner("🔮 Пиксель думает..."):
                # 5. Промпт Инжиниринг
                system_prompt = f"""
Ты — умный ассистент "Пиксель" по базе данных Disney. 

ТВОИ ДАННЫЕ (Контекст):
{knowledge_base_text}

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе предоставленных данных. Не выдумывай факты.
2. СТРОГО различай "Фильмы" (Live-action) и "Мультфильмы" (Animation). Смотри на поле "Тип" или "Жанр".
   - Если пользователь спрашивает про мультфильмы, не перечисляй кинофильмы.
   - Если спрашивает про фильмы, не перечисляй анимацию.
3. Если данных нет, отвечай: "К сожалению, эта информация не найдена в архиве."

ФОРМАТ ОТВЕТА (Строго соблюдай теги):
[РАССУЖДЕНИЕ]
Здесь напиши свой ход мыслей: как ты искал информацию, как фильтровал по типу (мультфильм/фильм), какие записи нашел.
[ОТВЕТ]
Здесь напиши финальный ответ для пользователя в вежливой форме.
"""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1, # Низкая температура для точности
                    max_tokens=2000
                )

                full_response = response.choices[0].message.content

                # 6. Парсинг ответа (Рассуждение vs Ответ)
                reasoning = ""
                final_answer = full_response

                if "[РАССУЖДЕНИЕ]" in full_response and "[ОТВЕТ]" in full_response:
                    parts = full_response.split("[ОТВЕТ]")
                    reasoning = parts[0].replace("[РАССУЖДЕНИЕ]", "").strip()
                    final_answer = parts[1].strip()
                
                # Вывод рассуждений в скрытом блоке
                if reasoning:
                    with st.expander("🕵️ Показать ход мыслей (Анализ)"):
                        st.write(reasoning)

                # Вывод финального ответа
                st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Произошла ошибка: {e}</div>', unsafe_allow_html=True)
