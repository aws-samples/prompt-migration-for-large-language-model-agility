
class TextSplitter:
    def __init__(self, chunk_avg_size, section_marker="##"):
        self.chunk_avg_size = chunk_avg_size
        self.section_marker = section_marker

    def split_txt(self, text):
        sections = text.split(self.section_marker)
        chunks = []
        chunk = ""
        for section in sections:
            chunk += section

            # don't break on small sections
            if len(section) < (self.chunk_avg_size // 5):
                continue

            # check if the chunk is complete
            if len(chunk) >= self.chunk_avg_size:
                chunks.append(chunk)
                chunk = ""

        if chunk != "":
            chunks.append(chunk)
        return chunks