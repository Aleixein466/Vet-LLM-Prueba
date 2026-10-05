"""Paso 2: dataset reducido veterinario en español (sintético) + pares SFT + preferencias DPO."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

ESPECIES = {
    "perro": {"art": "el", "plural": "perros", "peso": "5 a 40 kg", "fc": "60 a 140 lpm", "t": "38.3 a 39.2 °C"},
    "gato": {"art": "el", "plural": "gatos", "peso": "2.5 a 6 kg", "fc": "140 a 220 lpm", "t": "38.0 a 39.2 °C"},
    "caballo": {"art": "el", "plural": "caballos", "peso": "380 a 550 kg", "fc": "28 a 44 lpm", "t": "37.5 a 38.5 °C"},
    "vaca": {"art": "la", "plural": "vacas", "peso": "400 a 700 kg", "fc": "60 a 80 lpm", "t": "38.0 a 39.0 °C"},
    "conejo": {"art": "el", "plural": "conejos", "peso": "1 a 3 kg", "fc": "180 a 350 lpm", "t": "38.5 a 40.0 °C"},
}

TEMAS = [
    "vacunación {esp} cachorro: polivalente a las 6-8 semanas, refuerzo a las 3-4 semanas y rabia según normativa. Registrar fecha, lote y próxima dosis.",
    "desparasitación interna en {pl}: plan cada 3 meses con antihelmíntico de amplio espectro. En cachorros, cada 2 semanas hasta los 3 meses.",
    "control de pulgas y garrapatas en {pl}: tratamiento tópico o comprimido mensual, revisar orejas, cuello e ingle. Lavar cama a 60 °C.",
    "alimentación de {art} {esp}: ración según peso ({peso}), agua fresca siempre. Evitar cebolla, ajo, chocolate, xilitol y uvas.",
    "signos de alarma en {pl}: decaimiento, anorexia más de 24 h, vómitos repetidos, diarrea con sangre, dificultad respiratoria o fiebre ({t}). Acudir a clínica.",
    "constantes normales en {art} {esp}: frecuencia cardiaca {fc}, temperatura {t}. Medir en reposo y anotar en la ficha clínica.",
    "higiene dental en {pl}: cepillado 2-3 veces por semana, snacks dentales y revisión anual. El sarro avanzado requiere limpieza bajo anestesia.",
    "esterilización en {art} {esp}: reduce camadas no deseadas y ciertos tumores. Ayuno previo de 8 h, control postquirúrgico de la herida 10 días.",
    "manejo del estrés en {art} {esp}: rutina estable, enriquecimiento ambiental, transporte en transportín ventilado. Evitar castigos físicos.",
    "primeros auxilios en {art} {esp}: ante herida, presionar con gasa limpia; ante golpe de calor, enfriar con paños húmedos y acudir urgente.",
    "calendario de revisión en {pl}: cachorros cada 3-4 semanas, adultos una vez al año, geriátricos cada 6 meses con analítica sanguínea.",
    "parvovirosis en {pl}: vómitos, diarrea hemorrágica y deshidratación grave en cachorros no vacunados. Hospitalización con fluidoterapia.",
    "rinotraqueítis felina en {pl}: estornudos, secreción ocular y fiebre. Aislar, humidificar ambiente y seguir antibiótico si hay sobreinfección.",
    "cólico en {art} {esp}: inquietud, mirarse los flancos, sudoración. No dejar revolcarse sin control, llamar al veterinario de inmediato.",
    "mastitis en {art} {esp}: ubre caliente y dolorida, leche alterada. Ordeño frecuente, antiinflamatorio y antibiótico según antibiograma.",
]

QA_BASE = [
    ("desparasito", "¿Cada cuánto desparasito a mi {esp}?",
     "Cada 3 meses en adultos y cada 2 semanas hasta los 3 meses en cachorros, con producto adecuado al peso ({peso})."),
    ("vacunas", "¿Qué vacunas necesita un {esp} cachorro?",
     "Polivalente a las 6-8 semanas con refuerzo a las 3-4 semanas, más rabia según normativa local. Guarda la cartilla con lote y fecha."),
    ("urgencias", "¿Cuándo debo llevar a {art} {esp} a urgencias?",
     "Si hay dificultad respiratoria, sangrado que no cede, convulsiones, abdomen hinchado o decaimiento grave. No esperes 24 horas."),
    ("alimentación", "¿Qué no debe comer un {esp}?",
     "Chocolate, cebolla, ajo, uvas, xilitol y huesos cocidos. Mantén ración según peso ({peso}) y agua fresca siempre."),
    ("fiebre", "¿Cómo sé si {art} {esp} tiene fiebre?",
     "La temperatura normal es {t}. Usa termómetro rectal con vaselina; si supera el rango en reposo, consulta al veterinario."),
    ("vacunación", "¿Cada cuánto se vacuna un {esp} adulto?",
     "Tras el plan de cachorro, refuerzo anual según vacuna y normativa local. Anota fecha y lote en la cartilla."),
    ("pulgas", "¿Cómo elimino pulgas en {art} {esp}?",
     "Tratamiento mensual tópico o en comprimido, lava la cama a 60 grados y aspira la casa. Revisa orejas, cuello e ingle."),
    ("esterilización", "¿Es buena la esterilización en {art} {esp}?",
     "Sí, reduce camadas no deseadas y ciertos tumores. Requiere ayuno de 8 horas y control de la herida durante 10 días."),
]

RECHAZADAS = [
    "Cada 2 años con cualquier producto, el peso no importa.",
    "Dale chocolate y cebolla, son buenos para {pl}.",
    "No hace falta vacunar, los cachorros son inmunes solos.",
    "Si tiene fiebre, dale ibuprofeno humano sin consultar.",
    "Ante un cólico, deja que se revuelque solo y espera una semana.",
    "Las pulgas se quitan solo con agua, no uses antiparasitarios.",
]

EMERGENCIAS = [
    "intoxicación por chocolate en {art} {esp}: la teobromina causa vómitos, diarrea, agitación y convulsiones. Acude URGENTE al veterinario con la etiqueta del producto. No induzcas el vómito en casa sin indicación profesional ni des leche ni remedios caseros.",
    "intoxicación por xilitol en {pl}: provoca bajada brusca de glucosa y fallo hepático. Si sospechas ingesta (chicles, dulces sin azúcar), acude URGENTE al veterinario.",
    "ingesta de uvas o pasas en {art} {esp}: puede causar fallo renal agudo. Acude al veterinario aunque no haya síntomas todavía.",
    "intoxicación por cebolla o ajo en {pl}: destruye glóbulos rojos y causa anemia (debilidad, encías pálidas). Acude al veterinario.",
    "anticongelante en {pl}: sabor dulce pero mortal; el tratamiento solo funciona en las primeras horas. Urgencia máxima.",
    "raticida en {art} {esp}: provoca sangrados (encías, orina, heces). Guarda el envase y acude URGENTE; el antídoto (vitamina K) lo pauta el veterinario.",
    "convulsiones en {art} {esp}: no metas las manos en su boca, aparta objetos peligrosos, cronometra la crisis y acude al veterinario. Si dura más de 5 minutos es emergencia.",
    "atropello de {art} {esp}: muévelo lo mínimo, sobre superficie rígida, no des agua ni comida si está inconsciente y acude URGENTE.",
    "golpe de calor en {art} {esp}: jadeo extremo, babeo y tambaleo. Lleva a la sombra, enfría con paños húmedos (no hielo) y acude URGENTE.",
    "torsión gástrica en {art} {esp} (razas grandes): abdomen hinchado y arcadas sin vómito. Urgencia quirúrgica: acude de inmediato.",
    "hemorragia en {art} {esp}: presiona con gasa limpia 10 minutos sin levantar para mirar. No uses torniquete salvo indicación veterinaria. Acude a clínica.",
    "sospecha de envenenamiento en {pl}: recoge posible resto del producto, no provoques el vómito por tu cuenta y acude URGENTE al veterinario.",
]

QA_EMERG = [
    ("chocolate", "Mi {esp} comió chocolate, ¿qué hago?",
     "Acude de inmediato al veterinario con la etiqueta del producto. No induzcas el vómito en casa sin indicación profesional ni le des leche ni remedios caseros. Vigila vómitos, agitación o convulsiones."),
    ("convulsiones", "¿Qué hago si {art} {esp} tiene convulsiones?",
     "No metas las manos en su boca, aparta objetos con los que pueda golpearse, cronometra la crisis y acude al veterinario. Si supera 5 minutos es emergencia."),
    ("golpe de calor", "¿Cómo actúo ante un golpe de calor en {art} {esp}?",
     "Llévalo a la sombra, enfría con paños húmedos (nunca hielo) y acude URGENTE al veterinario. Jadeo extremo y tambaleo son signos graves."),
    ("sangrado", "Mi {esp} sangra mucho de una pata, ¿qué hago?",
     "Presiona con gasa limpia durante 10 minutos sin levantar para mirar. No uses torniquete salvo indicación veterinaria y acude a clínica."),
    ("envenenamiento", "¿Qué hago si sospecho que {art} {esp} se ha envenenado?",
     "Recoge restos del posible producto, no provoques el vómito por tu cuenta y acude URGENTE al veterinario con el envase si lo tienes."),
    ("urgencia cojera", "¿Cuándo una cojera de {art} {esp} es urgencia?",
     "Si no apoya la pata, hay inflamación grande, llanto de dolor o fue tras un golpe o atropello. Si solo cojea leve y apoya, pide cita pronto."),
]

PARAFRASIS_Q = [
    "{q}",
    "{q} Gracias.",
    "Hola, {ql} Por favor.",
    "Duda: {ql}",
]

def generar_docs(n_docs: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    docs: list[str] = []
    especies = list(ESPECIES)
    for _ in range(n_docs):
        esp = rng.choice(especies)
        info = ESPECIES[esp]
        tema = rng.choice(TEMAS)
        docs.append("Veterinaria %s. %s" % (esp, tema.format(
            esp=esp, pl=info["plural"], art=info["art"], peso=info["peso"], fc=info["fc"], t=info["t"])))
        if rng.random() < 0.35:
            _, q, a = rng.choice(QA_BASE)
            docs.append("Pregunta: %s\nRespuesta: %s" % (
                q.format(esp=esp, art=info["art"]),
                a.format(peso=info["peso"], t=info["t"])))
        if rng.random() < 0.20:
            tema = rng.choice(EMERGENCIAS)
            docs.append("Urgencia veterinaria %s. %s" % (esp, tema.format(
                esp=esp, pl=info["plural"], art=info["art"])))
            if rng.random() < 0.5:
                _, q, a = rng.choice(QA_EMERG)
                docs.append("Pregunta: %s\nRespuesta: %s" % (
                    q.format(esp=esp, art=info["art"]), a.format(esp=esp, art=info["art"])))
    rng.shuffle(docs)
    return docs

def generar_sft(seed: int) -> list[dict]:
    rng = random.Random(seed)
    pares: list[dict] = []
    for esp, info in ESPECIES.items():
        for _, q, a in list(QA_BASE) + list(QA_EMERG):
            for pf in PARAFRASIS_Q:
                qq = q.format(esp=esp, art=info["art"])
                ql = qq[0].lower() + qq[1:]
                pares.append({
                    "prompt": "Pregunta: %s\nRespuesta:" % pf.format(q=qq, ql=ql),
                    "response": " %s" % a.format(esp=esp, art=info["art"], peso=info["peso"], t=info["t"]),
                })
    rng.shuffle(pares)
    return pares  # 5 especies x 14 QA x 4 = 280

def generar_prefs(sft: list[dict], seed: int, n: int = 120) -> list[dict]:
    rng = random.Random(seed + 1)
    prefs = []
    for p in rng.sample(sft, min(n, len(sft))):
        rej = rng.choice(RECHAZADAS)
        if "{pl}" in rej:
            rej = rej.replace("{pl}", "mascotas")
        prefs.append({"prompt": p["prompt"], "chosen": p["response"], "rejected": " " + rej})
    return prefs

def main(n_docs=4000, seed=42):
    docs = generar_docs(n_docs, seed)
    raw = Path("data/raw/vet-es-sintetico.txt")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("\n\n".join(docs), encoding="utf-8")
    print(f"docs={len(docs)} chars={sum(map(len, docs))} -> {raw}")

    sft = generar_sft(seed)
    cut = int(len(sft) * 0.85)
    d = Path("data/sft")
    d.mkdir(parents=True, exist_ok=True)
    (d / "sft_train.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in sft[:cut]), encoding="utf-8")
    (d / "sft_val.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in sft[cut:]), encoding="utf-8")
    prefs = generar_prefs(sft[:cut], seed)
    (d / "prefs.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in prefs), encoding="utf-8")
    print(f"sft_train={cut} sft_val={len(sft)-cut} prefs={len(prefs)} -> {d}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.n_docs, a.seed)
