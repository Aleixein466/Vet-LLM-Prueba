"""Bateria controlada RAG: MODO A (sin RAG) vs MODO B (con RAG) + 6 tests.
Genera outputs/evaluation/rag_tests.json y rag_tests.txt
Uso: C:\\VET-TINY-LLM-venv\\Scripts\\python.exe scripts/run_rag_tests.py
"""
import sys, json
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch, yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT
from src.rag.retriever import TfidfIndex
from app import vet_chat as vc

CFG = "configs/tiny-18m.yaml"
INDEX = "data/rag/index"
K = 2

c = yaml.safe_load(Path(CFG).read_text(encoding="utf-8"))
m, inf = c["model"], c["inference"]
enc = get_enc()
idx = TfidfIndex.load(Path(INDEX))
model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
sd = torch.load(Path(inf["checkpoint"]), map_location="cpu")
model.load_state_dict(sd["model"] if "model" in sd else sd)
model.eval()
print("modelo listo", flush=True)

def gen_plain(prompt_ids):
    x = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=inf["max_new_tokens"],
                             temperature=inf["temperature"], top_k=inf["top_k"],
                             top_p=inf["top_p"],
                             repetition_penalty=inf.get("repetition_penalty", 1.0))
    gen = out[0].tolist()[len(prompt_ids):]
    if EOT in gen:
        gen = gen[:gen.index(EOT)]
    return enc.decode(gen).strip()

def modo_a(query):
    ids = enc.encode(f"Pregunta: {query}\nRespuesta:")
    if len(ids) > m["block_size"] - 1:
        ids = ids[-(m["block_size"] - 1):]
    return gen_plain(ids), "GENERATIVO_SIN_RAG"

def modo_b(query):
    info = vc.retrieve_debug(idx, query, k=K)
    if info["decision"] != "RECUPERA":
        # Paridad con la app (vet_chat.answer): respaldo semántico con DOBLE
        # puerta score+solape; sin esto Colón/dinosaurios colaban.
        sem = vc.retrieve_semantic(query, k=K, index=INDEX)
        n_over = len(info.get("overlap_vocab", []))
        sem_ok = False
        if sem:
            s_top = sem[0][1]
            sem_ok = ((s_top >= vc.SEM_FALLBACK_MIN and n_over >= vc.SEM_FALLBACK_MIN_OVERLAP)
                      or (s_top >= vc.SEM_FALLBACK_HIGH and n_over >= vc.SEM_FALLBACK_HIGH_OVERLAP))
        if sem_ok:
            info = {**info, "decision": "RECUPERA", "semantic_fallback": True,
                    "selected": [{"doc": i, "raw_score": round(s, 4), "bonus": 0.0,
                                  "score": round(s, 4), "shared": [],
                                  "n_shared": 0} for i, s, _ in sem]}
        else:
            msg = vc.FUERA_DE_AMBITO if info["decision"] == "FUERA_DE_AMBITO" else vc.SIN_EVIDENCIA
            return msg, [], info, "RECHAZADO", None
    passages = [idx.docs[s["doc"]] for s in info["selected"]]
    ids = vc.build_prompt(enc, passages, query, m["block_size"], None)
    raw = gen_plain(ids)
    top = passages[0]
    kw_top = vc._kw(top)
    rec = len(vc._kw(raw) & kw_top) / max(1, len(kw_top))
    if rec < 0.3:
        text = top.split("Respuesta:", 1)[1].strip() if "Respuesta:" in top else top
        text += " [extractivo]"
        mode = "EXTRACTIVO"
    else:
        text, mode = raw, "GENERATIVO_RAG"
    if vc.es_urgencia(query):
        text = vc.BANNER + "\n\n" + text
    return text, passages, info, mode, round(rec, 4)

QUERIES_MODO = [
    "mi perro lo golpeo un carro en la pata y lo veo agitado",
    "cristobal colon comio perros",
    "los dinosaurios comian perros?",
    "¿qué es la ehrlichiosis canina?",
    "¿qué signos tiene un perro con ehrlichiosis?",
]

TESTS = [
    {"id": "TEST 1", "query": "¿Qué signos presenta un perro con ehrlichiosis?",
     "tipo": "veterinaria", "esperado": "RECUPERAR información sobre ehrlichiosis/signos (doc con 'signos')",
     "check": lambda info: info["decision"] == "RECUPERA" and any("signos" in d.lower() for d in [idx.docs[s["doc"]] for s in info["selected"]])},
    {"id": "TEST 2", "query": "¿Qué información se debe recopilar en un perro con vómito?",
     "tipo": "veterinaria", "esperado": "RECUPERAR información sobre vómito/historia clínica",
     "check": lambda info: info["decision"] == "RECUPERA"},
    {"id": "TEST 3", "query": "¿Cuál es la capital de Francia?",
     "tipo": "fuera_del_dominio", "esperado": "NO recuperar (RECHAZADO), nada de envenenamiento",
     "check": lambda info: info["decision"] in ("FUERA_DE_AMBITO", "SIN_EVIDENCIA")},
    {"id": "TEST 4", "query": "¿Los dinosaurios comían perros?",
     "tipo": "fuera_del_dominio", "esperado": "NO recuperar envenenamiento (RECHAZADO)",
     "check": lambda info: info["decision"] in ("FUERA_DE_AMBITO", "SIN_EVIDENCIA")},
    {"id": "TEST 5", "query": "¿Quién fue Cristóbal Colón?",
     "tipo": "historia", "esperado": "NO recuperar información veterinaria (RECHAZADO)",
     "check": lambda info: info["decision"] in ("FUERA_DE_AMBITO", "SIN_EVIDENCIA")},
    {"id": "TEST 6", "query": "¿Qué es la ehrlichiosis canina?",
     "tipo": "veterinaria", "esperado": "RECUPERAR información relevante (doc con 'ehrlichiosis')",
     "check": lambda info: info["decision"] == "RECUPERA" and any("ehrlichiosis" in idx.docs[s["doc"]].lower() for s in info["selected"])},
    {"id": "TEST 7", "query": "mi can vomita sangre y tiene diarrea",
     "tipo": "veterinaria-parafrasis", "esperado": "RESCATE semántico (TF-IDF rechaza, MiniLM rescata parvovirosis)",
     "check": lambda info: info.get("semantic_fallback") is True},
]

res = {"fecha": datetime.now(timezone.utc).isoformat(), "cfg": CFG,
       "checkpoint": inf["checkpoint"], "index": INDEX, "top_k": K,
       "threshold": vc.SIM_THRESHOLD, "min_shared_terms": vc.MIN_SHARED_TERMS,
       "embedding": f"TF-IDF numpy vocab={len(idx.vocab)} docs={len(idx.docs)} cosine",
       "modo_comparacion": [], "tests": []}

for q in QUERIES_MODO:
    rA, modeA = modo_a(q)
    outB = modo_b(q)
    textB, passagesB, infoB, modeB, recB = outB
    res["modo_comparacion"].append({
        "query": q,
        "modo_a_sin_rag": {"mode": modeA, "respuesta": rA},
        "modo_b_con_rag": {"mode": modeB, "decision": infoB["decision"],
                           "threshold": infoB["threshold"],
                           "candidatos": infoB["candidates"][:4],
                           "seleccionados": infoB["selected"],
                           "rec_lexico": recB,
                           "respuesta": textB,
                           "pasajes": [p[:220] for p in passagesB]}})
    print(f"OK modos: {q[:50]!r} A={modeA} B={modeB}/{infoB['decision']}", flush=True)

npas = nfai = 0
for t in TESTS:
    outB = modo_b(t["query"])
    textB, passagesB, infoB, modeB, recB = outB
    ok = bool(t["check"](infoB))
    npas += ok
    nfai += (not ok)
    res["tests"].append({
        "id": t["id"], "query": t["query"], "tipo": t["tipo"],
        "top_k": K, "threshold": infoB["threshold"],
        "min_shared_terms": infoB["min_shared_terms"],
        "scores": [{"doc": s["doc"], "score": s["score"], "raw": s["raw_score"],
                    "shared": s["shared"]} for s in infoB["selected"]],
        "candidatos_top": infoB["candidates"][:4],
        "documento_recuperado": (idx.docs[infoB["selected"][0]["doc"]][:300] if infoB["selected"] else None),
        "modo_respuesta": modeB, "decision": infoB["decision"],
        "respuesta_final": textB,
        "esperado": t["esperado"],
        "obtenido": f"{infoB['decision']}/{modeB}",
        "result": "PASS" if ok else "FAIL"})
    print(f"{t['id']} {t['query'][:40]!r} -> {infoB['decision']}/{modeB} {'PASS' if ok else 'FAIL'}", flush=True)

res["resumen"] = {"pass": npas, "fail": nfai, "total": len(TESTS)}
outdir = Path("outputs/evaluation")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "rag_tests.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [f"VET-TINY-GPT RAG tests {res['fecha']}", f"threshold={res['threshold']} min_shared={res['min_shared_terms']} k={K}",
         f"embedding: {res['embedding']}", f"checkpoint: {res['checkpoint']}", "",
         "== MODO A (sin RAG) vs MODO B (con RAG) =="]
for e in res["modo_comparacion"]:
    lines += [f"--- {e['query']}", f"[A:{e['modo_a_sin_rag']['mode']}] {e['modo_a_sin_rag']['respuesta'][:250]}",
              f"[B:{e['modo_b_con_rag']['mode']}/{e['modo_b_con_rag']['decision']}] {e['modo_b_con_rag']['respuesta'][:250]}", ""]
lines.append("== 6 TESTS ==")
for t in res["tests"]:
    lines += [f"{t['id']} [{t['result']}] {t['query']}", f"  esperado: {t['esperado']}",
              f"  obtenido: {t['obtenido']}", f"  scores: {t['scores']}",
              f"  respuesta: {t['respuesta_final'][:250]}", ""]
lines.append(f"RESUMEN: {npas} PASS / {nfai} FAIL de {len(TESTS)}")
(outdir / "rag_tests.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"guardado rag_tests.json/txt: {npas} PASS {nfai} FAIL")
