import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import date, timedelta
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

st.set_page_config(page_title="Acompanhamento de Semaglutida", page_icon="📉", layout="centered")

# --- 1. CONFIGURAÇÃO DO SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except KeyError as error:
    raise RuntimeError(
        "Configure SUPABASE_URL e SUPABASE_KEY em .streamlit/secrets.toml."
    ) from error

APP_URL = st.secrets.get("APP_URL", "").strip()


class StreamlitAuthStorage:
    def get_item(self, key):
        return st.session_state.get(f"supabase_auth_{key}")

    def set_item(self, key, value):
        st.session_state[f"supabase_auth_{key}"] = value
        if key.endswith("-code-verifier"):
            st.session_state.supabase_pkce_verifier = value

    def remove_item(self, key):
        st.session_state.pop(f"supabase_auth_{key}", None)


def get_supabase_client():
    if "supabase_client" not in st.session_state:
        options = SyncClientOptions(
            storage=StreamlitAuthStorage(),
            flow_type="pkce",
        )
        st.session_state.supabase_client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options,
        )
    return st.session_state.supabase_client

supabase = get_supabase_client()


def get_authenticated_user():
    auth_code = st.query_params.get("code")
    if auth_code:
        code_verifier = st.query_params.get("pkce_verifier")
        exchange_params = {"auth_code": auth_code}
        if code_verifier:
            exchange_params["code_verifier"] = code_verifier
        try:
            supabase.auth.exchange_code_for_session(exchange_params)
            st.query_params.clear()
        except Exception as error:
            if "code verifier" in str(error).lower():
                    st.error("A sessão de login expirou. Clique em Entrar com Google novamente.")
            else:
                st.error("Não foi possível concluir o login com Google. Tente novamente.")
            st.stop()

    session = supabase.auth.get_session()
    return session.user if session else None


def login_with_google():
    if not APP_URL:
        st.error("Configure APP_URL com a URL pública do Streamlit Cloud nos secrets.")
        st.stop()

    credentials = {"provider": "google"}
    credentials["options"] = {"redirect_to": APP_URL}
    try:
        response = supabase.auth.sign_in_with_oauth(credentials)
    except Exception as error:
        if "provider is not enabled" in str(error).lower():
            st.error(
                "O login Google ainda não está ativado no Supabase. "
                "Ative Authentication > Providers > Google e configure as credenciais OAuth."
            )
            st.stop()
        raise
    code_verifier = st.session_state.get("supabase_pkce_verifier")
    if not code_verifier:
        st.error("Não foi possível preparar a sessão segura do login. Tente novamente.")
        st.stop()

    authorization_url = urlsplit(response.url)
    authorization_query = parse_qs(authorization_url.query, keep_blank_values=True)
    redirect_url = authorization_query.get("redirect_to", [APP_URL])[0]
    redirect_parts = urlsplit(redirect_url)
    redirect_query = parse_qs(redirect_parts.query, keep_blank_values=True)
    redirect_query["pkce_verifier"] = [code_verifier]
    redirect_url = urlunsplit((
        redirect_parts.scheme,
        redirect_parts.netloc,
        redirect_parts.path,
        urlencode(redirect_query, doseq=True),
        redirect_parts.fragment,
    ))
    authorization_query["redirect_to"] = [redirect_url]
    authorization_url = urlunsplit((
        authorization_url.scheme,
        authorization_url.netloc,
        authorization_url.path,
        urlencode(authorization_query, doseq=True),
        authorization_url.fragment,
    ))
    st.link_button("Entrar com Google", authorization_url, use_container_width=True)

# --- 2. FUNÇÕES DE BASE DE DADOS ---
def obter_perfil(user_id):
    try:
        resposta = supabase.table('utilizadores').select('*').eq('id', user_id).execute()
    except Exception:
        st.error("Não foi possível consultar o perfil no Supabase.")
        st.info(
            "No SQL Editor do Supabase, confirme que a tabela public.utilizadores "
            "existe e execute database/migrate_google_auth.sql."
        )
        st.stop()
    return resposta.data[0] if resposta.data else None

def obter_historico(user_id):
    resposta = supabase.table('registos_diarios').select('*').eq('user_id', user_id).order('data_registo').execute()
    return pd.DataFrame(resposta.data)

def guardar_registo(user_id, data_registo, peso, tomou, dose):
    try:
        supabase.table('registos_diarios').upsert({
            'user_id': user_id,
            'data_registo': str(data_registo),
            'peso': float(peso),
            'tomou_dose': tomou,
            'quantidade_dose': float(dose) if tomou else 0.0
        }, on_conflict='user_id,data_registo').execute()
    except Exception:
        st.error("Não foi possível salvar o registro no Supabase.")
        st.info(
            "Confirme que a tabela registos_diarios possui uma restrição única "
            "para user_id e data_registo e que as permissões authenticated estão ativas."
        )
        st.stop()

# --- 3. MOTOR DE INTELIGÊNCIA ARTIFICIAL ---
def gerar_predicao_ml(df_historico, peso_inicial):
    df_historico = df_historico.copy().sort_values('data_registo')
    df_historico['data_registo'] = pd.to_datetime(df_historico['data_registo'])
    data_inicio = df_historico['data_registo'].min()
    df_historico['Dias_Tratamento'] = (df_historico['data_registo'] - data_inicio).dt.days
    
    X = df_historico[['Dias_Tratamento']]
    y = df_historico['peso']
    
    # Utilizando o modelo Ridge que validámos (evita overfitting)
    modelo = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=10.0))
    modelo.fit(X, y)
    
    ultimo_dia = df_historico['Dias_Tratamento'].max()
    dias_alvo = np.array([10, 20, 30])
    dias_futuros = pd.DataFrame({'Dias_Tratamento': ultimo_dia + dias_alvo})
    datas_futuras = df_historico['data_registo'].max() + pd.to_timedelta(dias_alvo, unit='D')
    predicoes = modelo.predict(dias_futuros)
    peso_atual = float(df_historico['peso'].iloc[-1])
    perda_atual = ((peso_inicial - peso_atual) / peso_inicial) * 100
    projecoes = {
        int(dias): {
            'peso': float(peso),
            'perda': float(((peso_inicial - peso) / peso_inicial) * 100),
        }
        for dias, peso in zip(dias_alvo, predicoes)
    }
    
    # Geração do Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_historico['data_registo'], y, color='#1f77b4', label='Peso real', zorder=5)
    ax.plot(df_historico['data_registo'], modelo.predict(X), color='gray', linestyle='--', alpha=0.6, label='Tendência ajustada')
    ax.plot(datas_futuras, predicoes, color='#d62728', linestyle='-', linewidth=2, marker='o', label='Projeções')
    
    ax.set_title("Evolução e projeção do peso")
    ax.set_ylabel("Peso (kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, peso_atual, perda_atual, projecoes


def gerar_grafico_doses(df_historico):
    df_doses = df_historico.copy()
    df_doses['data_registo'] = pd.to_datetime(df_doses['data_registo'])
    df_doses = df_doses[df_doses['tomou_dose'] & (df_doses['quantidade_dose'] > 0)]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(df_doses['data_registo'], df_doses['quantidade_dose'], color='#2ca02c', width=0.8)
    ax.set_title("Doses registradas ao longo do tratamento")
    ax.set_ylabel("Dose (mg)")
    ax.set_xlabel("Data")
    ax.grid(axis='y', alpha=0.25)
    fig.autofmt_xdate()
    return fig

# --- 4. INTERFACE DO UTILIZADOR (FRONTEND) ---
st.title("📉 Acompanhamento com IA - Semaglutida")

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
    st.warning("Complete seu perfil para começar.")
    metadata = user.user_metadata or {}
    nome_padrao = metadata.get("full_name") or metadata.get("name") or ""
    with st.form("onboarding"):
        nome = st.text_input("Nome completo", value=nome_padrao)
        nascimento = st.date_input("Data de nascimento", min_value=date(1940, 1, 1), max_value=date.today())
        sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])
        peso_ini = st.number_input("Peso inicial (kg)", min_value=30.0, max_value=250.0, step=0.1)

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
    st.subheader("📝 Registrar peso")
    with st.form("registo_diario"):
        col1, col2 = st.columns(2)
        data_input = col1.date_input("Data da medição", value=date.today())
        peso_input = col2.number_input("Peso (kg)", min_value=30.0, max_value=250.0, step=0.1, value=float(df['peso'].iloc[-1]) if not df.empty else perfil['peso_inicial'])

        tomou_remedio = st.checkbox("Tomei a dose de semaglutida neste dia")
        dose_input = st.selectbox("Dose aplicada (mg)", [0.25, 0.5, 1.0, 2.0, 2.4]) if tomou_remedio else 0.0

        if st.form_submit_button("Salvar registro"):
            guardar_registo(perfil['id'], data_input, peso_input, tomou_remedio, dose_input)
            st.success("Registro salvo! A IA está recalculando sua curva...")
            st.rerun()

    # --- ZONA DA INTELIGÊNCIA ARTIFICIAL ---
    st.divider()
    st.subheader("🧠 Análise preditiva e histórico")

    if len(df) > 3:
        fig, peso_atual, perda_atual, projecoes = gerar_predicao_ml(df, perfil['peso_inicial'])
        st.pyplot(fig)

        st.caption("Percentuais calculados em relação ao peso inicial informado no perfil.")
        st.caption("As projeções são estatísticas e não substituem orientação médica.")
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        col_met1.metric("Perda atual", f"{perda_atual:.1f}%", help=f"Peso atual: {peso_atual:.1f} kg")
        col_met2.metric("Previsão em 10 dias", f"{projecoes[10]['perda']:.1f}%", help=f"Peso projetado: {projecoes[10]['peso']:.1f} kg")
        col_met3.metric("Previsão em 20 dias", f"{projecoes[20]['perda']:.1f}%", help=f"Peso projetado: {projecoes[20]['peso']:.1f} kg")
        col_met4.metric("Previsão em 30 dias", f"{projecoes[30]['perda']:.1f}%", help=f"Peso projetado: {projecoes[30]['peso']:.1f} kg")

        st.subheader("💉 Histórico de doses")
        df_doses = df[df['tomou_dose'] & (df['quantidade_dose'] > 0)]
        if not df_doses.empty:
            st.pyplot(gerar_grafico_doses(df), clear_figure=True)
        else:
            st.info("Ainda não há doses registradas para exibir neste gráfico.")

        perda_30d = projecoes[30]['perda']
        if perda_30d > 15:
            st.info("🔵 Ritmo projetado acelerado (acima da referência clínica)")
        elif perda_30d >= 6:
            st.success("🟢 Ritmo projetado esperado (de acordo com a referência clínica)")
        else:
            st.warning("🟠 Ritmo projetado lento (abaixo da referência clínica)")
    else:
        st.info("Continue registrando seu peso por mais alguns dias para a IA calcular uma curva personalizada.")