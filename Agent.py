import ollama
import pandas as pd

MODELO = "llama3.2"
ARCHIVO = "GasApp_Datos_Limpios.csv"
df = pd.read_csv(ARCHIVO)


def _error_columna(columna):
    if columna not in df.columns:
        return f"La columna {columna} no existe. Columnas: {list(df.columns)}"
    return None


def estadisticas(columna: str) -> str:
    """Devuelve estadisticas descriptivas de una columna del CSV.

    Args:
        columna: Nombre exacto de la columna.

    Returns:
        Texto con las estadisticas de la columna.
    """
    return _error_columna(columna) or df[columna].describe().to_string()


def total(columna: str) -> str:
    """Devuelve la suma total de una columna numerica del CSV.

    Args:
        columna: Nombre exacto de la columna numerica.

    Returns:
        Texto con la suma de la columna.
    """
    error = _error_columna(columna)
    if error:
        return error
    if not pd.api.types.is_numeric_dtype(df[columna]):
        return f"La columna {columna} no es numerica."
    return str(df[columna].sum())


def promedio_por_grupo(grupo: str, valor: str) -> str:
    """Devuelve el promedio de una columna numerica agrupado por otra columna.

    Args:
        grupo: Nombre exacto de la columna que define los grupos.
        valor: Nombre exacto de la columna numerica a promediar.

    Returns:
        Texto con el promedio de cada grupo.
    """
    error = _error_columna(grupo) or _error_columna(valor)
    if error:
        return error
    if not pd.api.types.is_numeric_dtype(df[valor]):
        return f"La columna {valor} no es numerica."
    return df.groupby(grupo)[valor].mean().to_string()


HERRAMIENTAS = [estadisticas, total, promedio_por_grupo]
FUNCIONES = {f.__name__: f for f in HERRAMIENTAS}
SISTEMA = {
    "role": "system",
    "content": (
        "Sos un asistente de datos. Respondé en español. "
        f"Columnas y tipos del CSV: {dict(df.dtypes.astype(str))}. "
        "Para cualquier dato numérico usá las herramientas, no inventes. "
        "Si una columna no existe, decilo y no uses otra."
    ),
}


def responder(pregunta, mensajes):
    """Responde una pregunta y devuelve (texto, herramientas_usadas)."""
    mensajes.append({"role": "user", "content": pregunta})
    usadas = []
    respuesta = ollama.chat(model=MODELO, messages=mensajes, tools=HERRAMIENTAS)
    mensajes.append(respuesta.message)
    if respuesta.message.tool_calls:
        for llamada in respuesta.message.tool_calls:
            nombre = llamada.function.name
            args = llamada.function.arguments
            usadas.append(f"{nombre}({args})")
            if nombre in FUNCIONES:
                resultado = FUNCIONES[nombre](**args)
            else:
                resultado = f"La herramienta {nombre} no existe."
            mensajes.append(
                {"role": "tool", "tool_name": nombre, "content": str(resultado)}
            )
        respuesta = ollama.chat(model=MODELO, messages=mensajes)
        mensajes.append(respuesta.message)
    return respuesta.message.content, usadas


if __name__ == "__main__":
    mensajes = [SISTEMA]
    while True:
        pregunta = input("Vos: ")
        if pregunta.strip().lower() in ("salir", "exit"):
            break
        try:
            texto, usadas = responder(pregunta, mensajes)
        except Exception as e:
            print("Error:", e)
            continue
        for u in usadas:
            print("[herramienta]", u)
        print("Agente:", texto)
