import os

def load_local_html(file_path):
    try:
        # Open and read the local HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        return html_content
    except FileNotFoundError:
        print(f">> The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f">> An error occurred: {e}")
        return None
