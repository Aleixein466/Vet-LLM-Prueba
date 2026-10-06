"""Componente: historia clínica (apoyo educativo, sin diagnósticos definitivos)."""
import streamlit as st
from app.utils.generation import generar, ADVERTENCIA

ALARMAS = [
    "sangr", "convulsi", "no respira", "inconsciente", "hinchad", "vómito",
    "diarrea", "cojea", "fiebre", "chocolate", "veneno", "atropell"
]


def render(model):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">📋 Historia Clínica</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Apoyo educativo para estructurar casos clínicos veterinarios</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(ADVERTENCIA + " Apoyo educativo: no emite diagnósticos definitivos.")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🐾 Datos del Paciente")
        
        especie = st.selectbox("Especie", ["Perro", "Gato", "Bovino", "Equino", "Otra"], key="clin_especie")
        
        c1, c2 = st.columns(2)
        with c1:
            raza = st.text_input("Raza", placeholder="Ej: Labrador", key="clin_raza")
            edad = st.text_input("Edad", placeholder="Ej: 5 años", key="clin_edad")
            sexo = st.selectbox("Sexo", ["", "Macho", "Hembra", "Macho castrado", "Hembra esterilizada"], key="clin_sexo")
        with c2:
            peso = st.text_input("Peso (kg)", placeholder="Ej: 25", key="clin_peso")
            temp = st.text_input("Temperatura (°C)", placeholder="Ej: 39.2", key="clin_temp")
        
        motivo = st.text_area("Motivo de consulta *", placeholder="Describe el motivo principal...", key="clin_motivo", height=100)
    
    with col2:
        st.markdown("### 📝 Antecedentes y Signos")
        
        antec = st.text_area("Antecedentes", placeholder="Vacunas, enfermedades previas, cirugías...", key="clin_antec", height=100)
        signos = st.text_area("Signos clínicos", placeholder="Describe los signos observados...", key="clin_signos", height=100)
        
        c3, c4 = st.columns(2)
        with c3:
            fc = st.text_input("Frecuencia cardíaca (lpm)", placeholder="Ej: 120", key="clin_fc")
        with c4:
            fr = st.text_input("Frecuencia respiratoria (rpm)", placeholder="Ej: 30", key="clin_fr")
        
        lab = st.text_area("Resultados de laboratorio", placeholder="Hemograma, bioquímica, etc.", key="clin_lab", height=80)
        trat = st.text_area("Tratamientos previos", placeholder="Medicamentos, dosis, respuesta...", key="clin_trat", height=80)
    
    # Analyze button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analizar = st.button("🔍 ANALIZAR CASO", type="primary", use_container_width=True)
    
    if analizar:
        campos = {
            "Especie": especie, "Raza": raza, "Edad": edad, "Sexo": sexo,
            "Peso": peso, "Motivo": motivo, "Antecedentes": antec, "Signos": signos,
            "Temperatura": temp, "FC": fc, "FR": fr,
            "Laboratorio": lab, "Tratamientos": trat
        }
        
        faltan = [k for k, v in campos.items() if not v.strip()]
        presentes = [(k, v) for k, v in campos.items() if v.strip()]
        alarmas = sorted({a for a in ALARMAS if a in (motivo + " " + signos).lower()})
        
        # Results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumen", "⚠️ Alarmas", "🤖 Modelo", "📄 Exportar"])
        
        with tab1:
            if presentes:
                for k, v in presentes:
                    st.markdown(f"""
                    <div class="card" style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: var(--text-secondary); font-weight: 500;">{k}</span>
                        <span style="color: var(--text-primary); text-align: right; max-width: 70%;">{v}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No hay datos ingresados")
            
            if faltan:
                st.markdown("---")
                st.markdown("#### Información faltante")
                for f in faltan:
                    st.markdown(f'<span class="badge-exp badge-warning">{f}</span>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card" style="border-color: var(--color-accent); margin-top: 1rem;">
                    <div style="color: var(--color-accent); font-weight: 600;">✅ Todos los campos completados</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            if alarmas:
                st.markdown("""
                <div class="card" style="border-color: var(--color-danger); background: rgba(239, 68, 68, 0.05);">
                    <div style="color: var(--color-danger); font-weight: 600; margin-bottom: 0.5rem;">⚠️ Signos de alarma detectados</div>
                </div>
                """, unsafe_allow_html=True)
                for a in alarmas:
                    st.markdown(f'<span class="badge-exp badge-danger" style="margin: 0.25rem;">{a}</span>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card" style="border-color: var(--color-accent); background: rgba(16, 185, 129, 0.05);">
                    <div style="color: var(--color-accent); font-weight: 600;">✅ No se detectaron signos de alarma en el texto ingresado</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            if motivo.strip():
                st.markdown("#### Generación del modelo (experimental, sin validar)")
                with st.spinner("Generando respuesta..."):
                    r = generar(
                        model,
                        f"Pregunta sobre {especie.lower()}: {motivo}\nRespuesta:",
                        temperature=0.0,
                        max_new_tokens=120
                    )
                st.markdown(f"""
                <div class="card" style="white-space: pre-wrap; line-height: 1.7;">
                    {r}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Ingresa un motivo de consulta para generar la respuesta del modelo")
        
        with tab4:
            # Generate structured report
            report = f"""HISTORIA CLÍNICA VETERINARIA
================================

DATOS DEL PACIENTE
- Especie: {especie}
- Raza: {raza or 'No especificada'}
- Edad: {edad or 'No especificada'}
- Sexo: {sexo or 'No especificado'}
- Peso: {peso or 'No especificado'} kg
- Temperatura: {temp or 'No especificada'} °C
- FC: {fc or 'No especificada'} lpm
- FR: {fr or 'No especificada'} rpm

MOTIVO DE CONSULTA
{motivo or 'No especificado'}

ANTECEDENTES
{antec or 'No especificados'}

SIGNOS CLÍNICOS
{signos or 'No especificados'}

RESULTADOS DE LABORATORIO
{lab or 'No disponibles'}

TRATAMIENTOS PREVIOS
{trat or 'Ninguno'}

ALARMAS DETECTADAS
{', '.join(alarmas) if alarmas else 'Ninguna'}

---
Generado por VET-TINY-GPT - Solo uso educativo
No sustituye evaluación veterinaria profesional
"""
            st.download_button(
                "📥 Descargar historia clínica",
                report,
                file_name=f"historia_clinica_{especie}_{raza or 'sin_raza'}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.markdown("---")
            st.code(report, language=None)
        
        st.info("Posibles diagnósticos diferenciales y tratamiento los define el veterinario. Este módulo no inventa laboratorios, dosis ni antecedentes.")


# Legacy function for backward compatibility
def _legacy_render(model):
    """Mantiene compatibilidad con código anterior."""
    render(model)