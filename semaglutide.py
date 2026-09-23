import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from datetime import date, timedelta
from pathlib import Path
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
LOGO_PATH = Path(__file__).with_name("LOGO-IMAGE-2026.png")


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
    
    modelo_ridge = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=10.0))
    modelo_svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1),
    )
    modelo_ridge.fit(X, y)
    modelo_svr.fit(X, y)
    
    ultimo_dia = df_historico['Dias_Tratamento'].max()
    dias_alvo = np.array([10, 20, 30])
    dias_grafico = np.arange(ultimo_dia + 1, ultimo_dia + 31)
    datas_grafico = df_historico['data_registo'].max() + pd.to_timedelta(
        np.arange(1, 31), unit='D'
    )
    dias_grafico_df = pd.DataFrame({'Dias_Tratamento': dias_grafico})
    dias_alvo_df = pd.DataFrame({'Dias_Tratamento': ultimo_dia + dias_alvo})
    predicoes_ridge_grafico = modelo_ridge.predict(dias_grafico_df)
    predicoes_svr_grafico = modelo_svr.predict(dias_grafico_df)
    predicoes_ridge = modelo_ridge.predict(dias_alvo_df)
    predicoes_svr = modelo_svr.predict(dias_alvo_df)
    peso_atual = float(df_historico['peso'].iloc[-1])
    perda_atual = ((peso_inicial - peso_atual) / peso_inicial) * 100
    projecoes = {
        int(dias): {
            'peso': float(peso_ridge),
            'perda': float(((peso_inicial - peso_ridge) / peso_inicial) * 100),
            'peso_svr': float(peso_svr),
            'perda_svr': float(((peso_inicial - peso_svr) / peso_inicial) * 100),
        }
        for dias, peso_ridge, peso_svr in zip(dias_alvo, predicoes_ridge, predicoes_svr)
    }
    
    # Geração do Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_historico['data_registo'], y, color='black', label='Peso real', zorder=5)
    ax.plot(
        df_historico['data_registo'],
        modelo_ridge.predict(X),
        color='blue',
        linestyle='--',
        alpha=0.8,
        label='Ridge Polynomial',
    )
    ax.plot(
        df_historico['data_registo'],
        modelo_svr.predict(X),
        color='green',
        linestyle='--',
        alpha=0.8,
        label='SVR (RBF)',
    )
    ax.plot(datas_grafico, predicoes_ridge_grafico, color='blue', linestyle='--', linewidth=2)
    ax.plot(datas_grafico, predicoes_svr_grafico, color='green', linestyle='--', linewidth=2)
    ax.scatter(
        df_historico['data_registo'].max() + pd.to_timedelta(dias_alvo, unit='D'),
        predicoes_ridge,
        color='blue',
        zorder=5,
        label='Ridge: pontos de 10, 20 e 30 dias',
    )
    ax.scatter(
        df_historico['data_registo'].max() + pd.to_timedelta(dias_alvo, unit='D'),
        predicoes_svr,
        color='green',
        zorder=5,
        label='SVR: pontos de 10, 20 e 30 dias',
    )
    
    ax.set_title("Evolução e projeção do peso")
    ax.set_ylabel("Peso (kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    erros = {
        'ridge': mean_absolute_error(y, modelo_ridge.predict(X)),
        'svr': mean_absolute_error(y, modelo_svr.predict(X)),
    }
    return fig, peso_atual, perda_atual, projecoes, erros


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


def filtrar_historico(df_historico, chave="periodo_relatorio"):
    if df_historico.empty:
        return df_historico

    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    data_minima = dados['data_registo'].min().date()
    data_maxima = dados['data_registo'].max().date()
    filtro = st.selectbox(
        "Período",
        ["Todos os registros", "Últimos 7 dias", "Últimos 30 dias", "Últimos 90 dias", "Período personalizado"],
        key=chave,
    )

    if filtro == "Todos os registros":
        return dados
    if filtro == "Período personalizado":
        col_inicio, col_fim = st.columns(2)
        data_inicio = col_inicio.date_input("Data inicial", value=data_minima, min_value=data_minima, max_value=data_maxima, key=f"{chave}_inicio")
        data_fim = col_fim.date_input("Data final", value=data_maxima, min_value=data_minima, max_value=data_maxima, key=f"{chave}_fim")
        return dados[
            (dados['data_registo'].dt.date >= data_inicio)
            & (dados['data_registo'].dt.date <= data_fim)
        ] if data_inicio <= data_fim else dados.iloc[0:0]

    dias = {"Últimos 7 dias": 7, "Últimos 30 dias": 30, "Últimos 90 dias": 90}[filtro]
    data_inicio = max(data_minima, data_maxima - timedelta(days=dias - 1))
    return dados[dados['data_registo'].dt.date >= data_inicio]


def gerar_pdf_historico(df_historico, titulo, perfil=None, tipo='historico', peso_inicial=None):
    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    tabela = dados[['data_registo', 'peso', 'tomou_dose', 'quantidade_dose']].copy()
    tabela.columns = ['Data', 'Peso (kg)', 'Tomou dose', 'Dose (mg)']
    tabela['Data'] = tabela['Data'].dt.strftime('%d/%m/%Y')
    tabela['Peso (kg)'] = tabela['Peso (kg)'].map(lambda valor: f'{valor:.1f}')
    tabela['Dose (mg)'] = tabela['Dose (mg)'].map(lambda valor: f'{valor:.2f}')

    def adicionar_marca(fig):
        if LOGO_PATH.exists():
            logo_ax = fig.add_axes([0.03, 0.02, 0.08, 0.05])
            logo_ax.imshow(plt.imread(LOGO_PATH))
            logo_ax.axis('off')
        fig.text(0.13, 0.035, 'Desenvolvido por Reinaldo Galvão', fontsize=8, color='#555555')

    def salvar_grafico_a4(pdf, figura):
        imagem = BytesIO()
        figura.savefig(imagem, format='png', dpi=150, bbox_inches='tight')
        imagem.seek(0)
        pagina, eixo = plt.subplots(figsize=(11.69, 8.27))
        eixo.set_position([0.06, 0.10, 0.88, 0.80])
        eixo.imshow(plt.imread(imagem), aspect='auto')
        eixo.axis('off')
        adicionar_marca(pagina)
        pdf.savefig(pagina)
        plt.close(pagina)
        plt.close(figura)

    arquivo = BytesIO()
    with PdfPages(arquivo) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.03, 0.92, titulo, fontsize=20, fontweight='bold')
        ax.text(0.03, 0.87, f'Gerado em {date.today().strftime("%d/%m/%Y")}', fontsize=10)
        ax.text(0.03, 0.80, f'Registros: {len(dados)}', fontsize=12)
        ax.text(0.03, 0.76, f'Peso médio: {dados["peso"].mean():.1f} kg', fontsize=12)
        ax.text(0.03, 0.72, f'Dose total: {dados.loc[dados["tomou_dose"], "quantidade_dose"].sum():.2f} mg', fontsize=12)
        if perfil:
            nascimento = pd.to_datetime(perfil['data_nascimento']).strftime('%d/%m/%Y')
            ax.text(0.03, 0.64, f'Nome: {perfil["nome"]}', fontsize=11)
            ax.text(0.03, 0.60, f'Sexo: {perfil["sexo"]}', fontsize=11)
            ax.text(0.03, 0.56, f'Data de nascimento: {nascimento}', fontsize=11)
        adicionar_marca(fig)
        pdf.savefig(fig)
        plt.close(fig)

        if tipo == 'acompanhamento' and len(dados) > 3:
            figura, _, _, _, _ = gerar_predicao_ml(dados, peso_inicial)
            salvar_grafico_a4(pdf, figura)
        elif tipo == 'estatisticas':
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.16)
            ax.plot(dados['data_registo'], dados['peso'], marker='o', color='#1f77b4', label='Peso registrado')
            if len(dados) >= 3:
                ax.plot(dados['data_registo'], dados['peso'].rolling(3, min_periods=1).mean(), color='#ff7f0e', linewidth=2, label='Média móvel (3 registros)')
            ax.set_title('Evolução do peso no período')
            ax.set_ylabel('Peso (kg)')
            ax.set_xlabel('Data')
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
            salvar_grafico_a4(pdf, fig)

            dados_doses = dados[dados['tomou_dose'] & (dados['quantidade_dose'] > 0)]
            if not dados_doses.empty:
                figura_doses = gerar_grafico_doses(dados)
                salvar_grafico_a4(pdf, figura_doses)
        else:
            for inicio in range(0, len(tabela), 25):
                pagina = tabela.iloc[inicio:inicio + 25]
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis('off')
                tabela_pdf = ax.table(
                    cellText=pagina.values,
                    colLabels=pagina.columns,
                    loc='center',
                    cellLoc='center',
                )
                tabela_pdf.auto_set_font_size(False)
                tabela_pdf.set_fontsize(10)
                tabela_pdf.scale(1, 1.6)
                adicionar_marca(fig)
                pdf.savefig(fig)
                plt.close(fig)

    return arquivo.getvalue()


def exibir_download_pdf(df_historico, titulo, perfil, tipo='historico', peso_inicial=None):
    if not df_historico.empty:
        st.download_button(
            "Baixar relatório em PDF",
            gerar_pdf_historico(df_historico, titulo, perfil, tipo, peso_inicial),
            file_name="relatorio_semaglutida.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def exibir_relatorio(df_historico, perfil):
    st.header("📄 Histórico")
    if df_historico.empty:
        st.info("Ainda não há registros para o período selecionado.")
        return

    dados_filtrados = filtrar_historico(df_historico)

    if dados_filtrados.empty:
        st.info("Não há registros no período selecionado.")
        return

    doses = dados_filtrados.loc[dados_filtrados['tomou_dose'], 'quantidade_dose'].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", len(dados_filtrados))
    col2.metric("Peso médio", f"{dados_filtrados['peso'].mean():.1f} kg")
    col3.metric("Dose total", f"{doses:.2f} mg")

    tabela = dados_filtrados[
        ['data_registo', 'peso', 'tomou_dose', 'quantidade_dose']
    ].rename(columns={
        'data_registo': 'Data',
        'peso': 'Peso (kg)',
        'tomou_dose': 'Tomou dose',
        'quantidade_dose': 'Dose (mg)',
    })
    tabela['Data'] = tabela['Data'].dt.strftime('%d/%m/%Y')
    st.dataframe(tabela, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar relatório em CSV",
        tabela.to_csv(index=False).encode('utf-8-sig'),
        file_name="relatorio_semaglutida.csv",
        mime="text/csv",
        use_container_width=True,
    )
    exibir_download_pdf(dados_filtrados, "Histórico de semaglutida", perfil, tipo='historico')


def exibir_estatisticas(df_historico, peso_inicial):
    st.header("📊 Estatísticas")
    if df_historico.empty:
        st.info("Ainda não há dados suficientes para calcular estatísticas.")
        return

    dados = filtrar_historico(df_historico, chave="periodo_estatisticas")
    if dados.empty:
        st.info("Não há registros no período selecionado.")
        return

    peso_atual = float(dados['peso'].iloc[-1])
    peso_minimo = float(dados['peso'].min())
    peso_maximo = float(dados['peso'].max())
    perda_periodo = ((float(dados['peso'].iloc[0]) - peso_atual) / float(dados['peso'].iloc[0])) * 100
    dias_com_dose = int(dados['tomou_dose'].sum())
    adesao = (dias_com_dose / len(dados)) * 100
    dose_media = float(dados.loc[dados['tomou_dose'], 'quantidade_dose'].mean()) if dias_com_dose else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", len(dados))
    col2.metric("Peso atual", f"{peso_atual:.1f} kg")
    col3.metric("Perda no período", f"{perda_periodo:.1f}%")
    col4, col5, col6 = st.columns(3)
    col4.metric("Menor peso", f"{peso_minimo:.1f} kg")
    col5.metric("Maior peso", f"{peso_maximo:.1f} kg")
    col6.metric("Adesão registrada", f"{adesao:.1f}%")
    st.caption(f"Dose média nos dias registrados: {dose_media:.2f} mg. Peso inicial do perfil: {peso_inicial:.1f} kg.")

    st.subheader("Evolução do peso no período")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dados['data_registo'], dados['peso'], marker='o', color='#1f77b4', label='Peso registrado')
    if len(dados) >= 3:
        ax.plot(dados['data_registo'], dados['peso'].rolling(3, min_periods=1).mean(), color='#ff7f0e', linewidth=2, label='Média móvel (3 registros)')
    ax.set_ylabel("Peso (kg)")
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Doses no período")
    dados_doses = dados[dados['tomou_dose'] & (dados['quantidade_dose'] > 0)]
    if dados_doses.empty:
        st.info("Não há doses registradas no período selecionado.")
    else:
        st.pyplot(gerar_grafico_doses(dados), clear_figure=True)
    exibir_download_pdf(dados, "Estatísticas do tratamento com semaglutida", perfil, tipo='estatisticas', peso_inicial=peso_inicial)

# --- 4. INTERFACE DO UTILIZADOR (FRONTEND) ---
st.title("📉 Acompanhamento com IA - Semaglutida")

user = get_authenticated_user()

if not user:
    st.info("Entre com a sua conta Google para continuar.")
    login_with_google()
    st.stop()

perfil = obter_perfil(user.id)

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=150)
    if perfil:
        nascimento_formatado = pd.to_datetime(perfil['data_nascimento']).strftime('%d/%m/%Y')
        st.markdown(f"**Nome:** {perfil['nome']}")
        st.markdown(f"**Sexo:** {perfil['sexo']}")
        st.markdown(f"**Data de nascimento:** {nascimento_formatado}")
    else:
        st.caption("Perfil ainda não preenchido")
    menu = st.radio("Menu", ["Acompanhamento", "Estatísticas", "Histórico"])
    st.caption("Desenvolvido por Reinaldo Galvão")
    if st.button("Sair", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.pop("supabase_client", None)
        st.rerun()

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

    if menu == "Histórico":
        exibir_relatorio(df, perfil)
        st.stop()
    if menu == "Estatísticas":
        exibir_estatisticas(df, perfil['peso_inicial'])
        st.stop()

    # --- ZONA DE REGISTO DIÁRIO ---
    st.subheader("📝 Registrar peso")
    form_version = st.session_state.get("registro_form_version", 0)
    with st.form(f"registro_diario_{form_version}"):
        col1, col2 = st.columns(2)
        data_input = col1.date_input("Data da medição", value=date.today(), key=f"data_registro_{form_version}")
        registro_do_dia = df[pd.to_datetime(df['data_registo']).dt.date == data_input] if not df.empty else df
        peso_padrao = float(registro_do_dia['peso'].iloc[0]) if not registro_do_dia.empty else (float(df['peso'].iloc[-1]) if not df.empty else perfil['peso_inicial'])
        tomou_padrao = bool(registro_do_dia['tomou_dose'].iloc[0]) if not registro_do_dia.empty else False
        dose_padrao = float(registro_do_dia['quantidade_dose'].iloc[0]) if not registro_do_dia.empty else 0.25
        opcoes_dose = [0.25, 0.5, 1.0, 2.0, 2.4]
        peso_input = col2.number_input("Peso (kg)", min_value=30.0, max_value=250.0, step=0.1, value=peso_padrao, key=f"peso_input_{form_version}_{data_input}")

        tomou_remedio = st.checkbox("Tomei a dose de semaglutida neste dia", value=tomou_padrao, key=f"tomou_remedio_{form_version}_{data_input}")
        dose_input = st.selectbox("Dose aplicada (mg)", opcoes_dose, index=opcoes_dose.index(dose_padrao) if dose_padrao in opcoes_dose else 0, key=f"dose_input_{form_version}_{data_input}") if tomou_remedio else 0.0

        col_salvar, col_cancelar = st.columns(2)
        salvar = col_salvar.form_submit_button("Salvar registro")
        cancelar = col_cancelar.form_submit_button("Cancelar")

        if cancelar:
            st.session_state.registro_form_version = form_version + 1
            st.rerun()
        if salvar:
            guardar_registo(perfil['id'], data_input, peso_input, tomou_remedio, dose_input)
            st.success("Registro salvo! A IA está recalculando sua curva...")
            st.rerun()

    if not df.empty:
        with st.expander("Ajustar registro existente"):
            datas_registradas = pd.to_datetime(df['data_registo']).dt.date.tolist()[::-1]
            data_ajuste = st.selectbox("Selecione a data para ajustar", datas_registradas)
            registro = df[pd.to_datetime(df['data_registo']).dt.date == data_ajuste].iloc[0]
            with st.form("ajuste_registro"):
                peso_ajuste = st.number_input(
                    "Peso registrado (kg)",
                    min_value=30.0,
                    max_value=250.0,
                    step=0.1,
                    value=float(registro['peso']),
                )
                tomou_ajuste = st.checkbox(
                    "Tomei a dose neste dia",
                    value=bool(registro['tomou_dose']),
                )
                dose_ajuste = st.selectbox(
                    "Dose aplicada (mg)",
                    [0.25, 0.5, 1.0, 2.0, 2.4],
                    index=[0.25, 0.5, 1.0, 2.0, 2.4].index(float(registro['quantidade_dose']))
                    if float(registro['quantidade_dose']) in [0.25, 0.5, 1.0, 2.0, 2.4]
                    else 0,
                ) if tomou_ajuste else 0.0
                if st.form_submit_button("Salvar ajuste"):
                    guardar_registo(perfil['id'], data_ajuste, peso_ajuste, tomou_ajuste, dose_ajuste)
                    st.success("Registro atualizado.")
                    st.rerun()

    # --- ZONA DA INTELIGÊNCIA ARTIFICIAL ---
    st.divider()
    st.subheader("🧠 Análise preditiva e histórico")

    if len(df) > 3:
        fig, peso_atual, perda_atual, projecoes, erros = gerar_predicao_ml(df, perfil['peso_inicial'])
        st.pyplot(fig)

        st.caption("Percentuais calculados em relação ao peso inicial informado no perfil.")
        st.caption("As projeções são estatísticas e não substituem orientação médica.")
        st.caption(f"Erro médio histórico — Ridge: {erros['ridge']:.3f} kg | SVR: {erros['svr']:.3f} kg")
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

    st.divider()
    st.caption("Relatório completo")
    exibir_download_pdf(df, "Acompanhamento de semaglutida", perfil, tipo='acompanhamento', peso_inicial=perfil['peso_inicial'])