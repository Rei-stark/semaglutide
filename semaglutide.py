import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import date, timedelta

st.set_page_config(page_title="Predição Semaglutida", page_icon="📉", layout="centered")

# --- 1. CONFIGURAÇÃO DO SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except KeyError as error:
    raise RuntimeError(
        "Configure SUPABASE_URL e SUPABASE_KEY em .streamlit/secrets.toml."
    ) from error

APP_URL = st.secrets.get("APP_URL", "").strip()

def get_supabase_client():
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return st.session_state.supabase_client

supabase = get_supabase_client()


def get_authenticated_user():
    auth_code = st.query_params.get("code")
    if auth_code:
        try:
            supabase.auth.exchange_code_for_session(auth_code)
            st.query_params.clear()
        except Exception:
            st.error("Não foi possível concluir o login com Google. Tente novamente.")
            st.stop()

    session = supabase.auth.get_session()
    return session.user if session else None


def login_with_google():
    credentials = {"provider": "google"}
    if APP_URL:
        credentials["options"] = {"redirect_to": APP_URL}
    response = supabase.auth.sign_in_with_oauth(credentials)
    st.link_button("Entrar com Google", response.url, use_container_width=True)

# --- 2. FUNÇÕES DE BASE DE DADOS ---
def obter_perfil(user_id):
    resposta = supabase.table('utilizadores').select('*').eq('id', user_id).maybe_single().execute()
    return resposta.data if resposta.data else None

def obter_historico(user_id):
    resposta = supabase.table('registos_diarios').select('*').eq('user_id', user_id).order('data_registo').execute()
    return pd.DataFrame(resposta.data)

def guardar_registo(user_id, data_registo, peso, tomou, dose):
    supabase.table('registos_diarios').upsert({
        'user_id': user_id,
        'data_registo': str(data_registo),
        'peso': float(peso),
        'tomou_dose': tomou,
        'quantidade_dose': float(dose) if tomou else 0.0
    }).execute()

# --- 3. MOTOR DE INTELIGÊNCIA ARTIFICIAL ---
def gerar_predicao_ml(df_historico, peso_inicial):
    df_historico['data_registo'] = pd.to_datetime(df_historico['data_registo'])
    data_inicio = df_historico['data_registo'].min()
    df_historico['Dias_Tratamento'] = (df_historico['data_registo'] - data_inicio).dt.days
    
    X = df_historico[['Dias_Tratamento']]
    y = df_historico['peso']
    
    # Utilizando o modelo Ridge que validámos (evita overfitting)
    modelo = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=10.0))
    modelo.fit(X, y)
    
    ultimo_dia = df_historico['Dias_Tratamento'].max()
    dias_futuros = pd.DataFrame({'Dias_Tratamento': np.arange(ultimo_dia + 1, ultimo_dia + 31)})
    datas_futuras = pd.date_range(start=df_historico['data_registo'].max() + timedelta(days=1), periods=30)
    
    predicao_futura = modelo.predict(dias_futuros)
    
    # Geração do Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_historico['data_registo'], y, color='#1f77b4', label='Peso Real Registado', zorder=5)
    ax.plot(df_historico['data_registo'], modelo.predict(X), color='gray', linestyle='--', alpha=0.6, label='Curva de Ajuste')
    ax.plot(datas_futuras, predicao_futura, color='#d62728', linestyle='-', linewidth=2, label='Predição (+30 Dias)')
    
    ax.set_title("Evolução e Predição de Emagrecimento")
    ax.set_ylabel("Peso (kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    peso_30d = predicao_futura[-1]
    perda_estimada = ((peso_inicial - peso_30d) / peso_inicial) * 100
    
    return fig, peso_30d, perda_estimada

# --- 4. INTERFACE DO UTILIZADOR (FRONTEND) ---
st.title("📉 Acompanhamento IA - Semaglutida")

user = get_authenticated_user()

if not user:
    st.info("Entre com a sua conta Google para continuar.")
    login_with_google()
    st.stop()

with st.sidebar:
    st.write(f"Conta: {user.email}")
    if st.button("Sair", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.pop("supabase_client", None)
        st.rerun()

perfil = obter_perfil(user.id)

if not perfil:
    st.warning("Complete o seu perfil para começar.")
    metadata = user.user_metadata or {}
    nome_padrao = metadata.get("full_name") or metadata.get("name") or ""
    with st.form("onboarding"):
        nome = st.text_input("Nome Completo", value=nome_padrao)
        nascimento = st.date_input("Data de Nascimento", min_value=date(1940, 1, 1), max_value=date.today())
        sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])
        peso_ini = st.number_input("Peso Inicial (kg)", min_value=30.0, max_value=250.0, step=0.1)

        if st.form_submit_button("Criar Perfil"):
            supabase.table('utilizadores').insert({
                'id': user.id, 'email': user.email, 'nome': nome,
                'data_nascimento': str(nascimento), 'sexo': sexo, 'peso_inicial': peso_ini
            }).execute()
            st.success("Perfil criado com sucesso.")
            st.rerun()
else:
    st.success(f"Olá, {perfil['nome']}! Bem-vindo de volta.")

    df = obter_historico(perfil['id'])

    # --- ZONA DE REGISTO DIÁRIO ---
    st.subheader("📝 Adicionar/Atualizar Peso")
    with st.form("registo_diario"):
        col1, col2 = st.columns(2)
        data_input = col1.date_input("Data da Medição", value=date.today())
        peso_input = col2.number_input("Peso (kg)", min_value=30.0, max_value=250.0, step=0.1, value=float(df['peso'].iloc[-1]) if not df.empty else perfil['peso_inicial'])

        tomou_remedio = st.checkbox("Tomei a dose de Semaglutida neste dia")
        dose_input = st.selectbox("Quantidade da dose (mg)", [0.25, 0.5, 1.0, 1.7, 2.4, 2.5]) if tomou_remedio else 0.0

        if st.form_submit_button("Guardar Registo"):
            guardar_registo(perfil['id'], data_input, peso_input, tomou_remedio, dose_input)
            st.success("Dados guardados! A IA está a recalcular a sua curva...")
            st.rerun()

    # --- ZONA DA INTELIGÊNCIA ARTIFICIAL ---
    st.divider()
    st.subheader("🧠 Análise Preditiva e Histórico")

    if len(df) > 3:
        fig, peso_projetado, perda = gerar_predicao_ml(df, perfil['peso_inicial'])
        st.pyplot(fig)

        col_met1, col_met2, col_met3 = st.columns(3)
        col_met1.metric("Peso Atual", f"{df['peso'].iloc[-1]:.1f} kg")
        col_met2.metric("Projeção (30 dias)", f"{peso_projetado:.1f} kg")
        col_met3.metric("Perda Total Estimada", f"{perda:.1f}%")

        if perda > 15:
            st.info("🔵 Ritmo Acelerado (Acima da média clínica)")
        elif perda >= 6:
            st.success("🟢 Ritmo Esperado (De acordo com a literatura médica)")
        else:
            st.warning("🟠 Ritmo Lento (Abaixo da média clínica)")
    else:
        st.info("Continue a registar o seu peso durante mais alguns dias para a Inteligência Artificial conseguir desenhar a sua curva personalizada.")