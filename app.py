from shiny import App, ui, render
from shinywidgets import output_widget, render_widget

from sklearn.linear_model import LinearRegression

import pandas as pd
import plotly.express as px

# ======================================
# LEER DATASET
# ======================================

df = pd.read_csv("Data/infocana_limpio.csv")

# ======================================
# MODELO DE REGRESION
# ======================================

X = df[["cana_molida_neta"]]
Y = df["azucar_producida_total"]

modelo = LinearRegression()
modelo.fit(X, Y)

# ======================================
# SERIE TEMPORAL
# ======================================

serie_zafra = df.groupby(
    "zafra"
)["azucar_producida_total"].sum().reset_index()

# ======================================
# KPIs
# ======================================

total_azucar = round(
    df["azucar_producida_total"].sum(),
    2
)

total_cana = round(
    df["cana_molida_neta"].sum(),
    2
)

promedio_rendimiento = round(
    df["rendimiento_agroindustrial"].mean(),
    2
)

# ======================================
# INTERFAZ
# ======================================

app_ui = ui.page_fluid(

    # =========================
    # CSS
    # =========================

    ui.tags.style("""

    body {
        background-color: #0f172a;
        color: white;
        font-family: Arial;
    }

    .titulo {
        color: #4ade80;
        font-weight: bold;
        text-shadow: 0px 0px 15px #22c55e;
    }

    .kpi-card {

        background: linear-gradient(
            145deg,
            #111827,
            #1e293b
        );

        padding: 25px;

        border-radius: 20px;

        text-align: center;

        box-shadow:
            0px 0px 20px rgba(34,197,94,0.2);

        transition: 0.3s;
    }

    .kpi-card:hover {

        transform: scale(1.03);

        box-shadow:
            0px 0px 30px rgba(34,197,94,0.6);
    }

    .kpi-card h2 {
        font-size: 42px;
        color: white;
    }

    .kpi-card h4 {
        color: #86efac;
    }

    .card {

        background-color: #111827 !important;

        border: 1px solid #22c55e !important;

        border-radius: 20px !important;

        box-shadow:
            0px 0px 15px rgba(34,197,94,0.15);
    }

    p, h2, h3 {
        color: white;
    }

    hr {
        border: 1px solid #22c55e;
    }

    select {

        background-color: #111827 !important;

        color: white !important;

        border: 1px solid #22c55e !important;
    }

    table {
        color: white !important;
    }

    """),

    # =========================
    # TITULO
    # =========================

    ui.h1(
        "📊 Dashboard Infocaña",
        class_="titulo"
    ),

    ui.p(
        "Análisis predictivo de producción "
        "de azúcar mediante regresión lineal"
    ),

    ui.hr(),

    # =========================
    # KPIs
    # =========================

    ui.row(

        ui.column(
            4,

            ui.div(

                ui.h4("🍬 Azúcar Total"),

                ui.h2(f"{total_azucar:,.0f}"),

                class_="kpi-card"
            )
        ),

        ui.column(
            4,

            ui.div(

                ui.h4("🚜 Caña Molida"),

                ui.h2(f"{total_cana:,.0f}"),

                class_="kpi-card"
            )
        ),

        ui.column(
            4,

            ui.div(

                ui.h4("📈 Rendimiento"),

                ui.h2(f"{promedio_rendimiento}%"),

                class_="kpi-card"
            )
        )
    ),

    ui.hr(),

    # =========================
    # SELECTOR
    # =========================

    ui.input_select(

        "variable",

        "Selecciona Variable:",

        {

            "cana_molida_neta":
                "Caña Molida Neta",

            "azucar_producida_total":
                "Azúcar Producida",

            "superficie_cosechada":
                "Superficie Cosechada"
        }
    ),

    ui.br(),

    # =========================
    # GRAFICAS
    # =========================

    ui.row(

        ui.column(
            6,

            ui.card(

                ui.card_header("📈 Tendencia"),

                output_widget("grafica_linea")
            )
        ),

        ui.column(
            6,

            ui.card(

                ui.card_header("📊 Distribución"),

                output_widget("grafica_histograma")
            )
        )
    ),

    ui.br(),

    # =========================
    # SERIE TEMPORAL
    # =========================

    ui.card(

        ui.card_header(
            "📅 Producción por Zafra"
        ),

        output_widget("grafica_zafra")
    ),

    ui.hr(),

    # =========================
    # PREDICCION
    # =========================

    ui.h2("🤖 Predicción Inteligente"),

    ui.input_numeric(

        "pred_input",

        "Cantidad de caña molida:",

        value=500000
    ),

    ui.output_text("prediccion"),

    ui.hr(),

    # =========================
    # TABLA
    # =========================

    ui.h2("📋 Tabla de datos"),

    ui.output_data_frame("tabla")
)

# ======================================
# SERVER
# ======================================

def server(input, output, session):

    # =========================
    # GRAFICA LINEA
    # =========================

    @output
    @render_widget
    def grafica_linea():

        fig = px.line(

            df,

            y=input.variable(),

            title="Serie Temporal"
        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#111827",

            plot_bgcolor="#111827",

            font=dict(color="white")
        )

        return fig

    # =========================
    # HISTOGRAMA
    # =========================

    @output
    @render_widget
    def grafica_histograma():

        fig = px.histogram(

            df,

            x=input.variable(),

            title="Distribución"
        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#111827",

            plot_bgcolor="#111827",

            font=dict(color="white")
        )

        return fig

    # =========================
    # GRAFICA ZAFRA
    # =========================

    @output
    @render_widget
    def grafica_zafra():

        fig = px.line(

            serie_zafra,

            x="zafra",

            y="azucar_producida_total",

            markers=True,

            title="Producción Total por Zafra"
        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#111827",

            plot_bgcolor="#111827",

            font=dict(color="white")
        )

        return fig

    # =========================
    # PREDICCION
    # =========================

    @output
    @render.text
    def prediccion():

        valor = input.pred_input()

        resultado = modelo.predict(
            [[valor]]
        )

        return (
            f"Producción estimada: "
            f"{resultado[0]:,.2f}"
        )

    # =========================
    # TABLA
    # =========================

    @output
    @render.data_frame
    def tabla():
        return df

# ======================================
# APP
# ======================================

app = App(app_ui, server)