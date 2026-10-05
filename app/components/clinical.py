"""Componente: historia clínica (apoyo educativo, sin diagnósticos definitivos)."""
import streamlit as st
from app.utils.generation import generar, ADVERTENCIA

ALARMAS = ["sangr", "convulsi", "no respira", "inconsciente", "hinchad", "vómito",
           "diarrea", "cojea", "fiebre", "chocolate", "veneno", "atropell"]


def render(model):
    st.header("📋 Historia clínica")
    st.caption(ADVERTENCIA + " Apoyo educativo: no emite diagnósticos definitivos.")
    c1, c2 = st.columns(2)
    with c1:
        especie = st.selectbox("Especie", ["Perro", "Gato", "Bovino", "Equino", "Otra"])
        raza = st.text_input("Raza")
        edad = st.text_input("Edad")
        sexo = st.selectbox("Sexo", ["", "Macho", "Hembra"])
        peso = st.text_input("Peso")
        motivo = st.text_area("Motivo de consulta")
    with c2:
        antec = st.text_area("Antecedentes")
        signos = st.text_area("Signos clínicos")
        temp = st.text_input("Temperatura")
        fc = st.text_input("Frecuencia cardíaca")
        fr = st.text_input("Frecuencia respiratoria")
        lab = st.text_area("Resultados de laboratorio")
        trat = st.text_area("Tratamientos previos")
    if st.button("ANALIZAR"):
        campos = {"Especie": especie, "Raza": raza, "Edad": edad, "Sexo": sexo,
                  "Peso": peso, "Motivo": motivo, "Antecedentes": antec, "Signos": signos,
                  "Temperatura": temp, "FC": fc, "FR": fr,
                  "Laboratorio": lab, "Tratamientos": trat}
        faltan = [k for k, v in campos.items() if not v.strip()]
        presentes = [f"**{k}**: {v}" for k, v in campos.items() if v.strip()]
        alarmas = sorted({a for a in ALARMAS if a in (motivo + " " + signos).lower()})
        st.subheader("Resumen")
        st.markdown("\n".join(f"- {p}" for p in presentes) or "Sin datos.")
        st.subheader("Información faltante")
        st.markdown(", ".join(faltan) if faltan else "Ninguna.")
        st.subheader("Signos de alarma")
        st.markdown(", ".join(alarmas) if alarmas else "Ninguno detectado en el texto.")
        if motivo.strip():
            st.subheader("Texto del modelo (experimental, sin validar)")
            r = generar(model, f"Pregunta sobre {especie.lower()}: {motivo}\nRespuesta:",
                        temperature=0.0, max_new_tokens=80)
            st.write(r)
        st.info("Posibles diagnósticos diferenciales y tratamiento los define el veterinario. "
                "Este módulo no inventa laboratorios, dosis ni antecedentes.")
