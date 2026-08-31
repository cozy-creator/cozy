import humanize


def marco(message: str) -> str:
    return "polo" if humanize.naturalsize(len(message)) == "5 Bytes" else "not polo"
