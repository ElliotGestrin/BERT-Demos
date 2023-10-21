import dearpygui.dearpygui as dpg
from src import BERT, FileManagement
import os
dpg.create_context()

def search():
    dpg.set_item_label("SearchButton","Getting BERT Model")
    BERT.new_bert(dpg.get_value("BERT Model"))
    dpg.set_item_label("SearchButton","Embedding Directory")
    dir_embs = FileManagement.embed_or_load_dir(dpg.get_value("Directory"))
    dpg.set_item_label("SearchButton","Searching Directory")
    doc_sims = BERT.term_similarity(
        pos_term=dpg.get_value("Positive Term"),
        neg_term=dpg.get_value("Negative Term"),
        embedded_documents=dir_embs,
        neg_weight=dpg.get_value("Negative Weight"),
        strategy=dpg.get_value("Method")
    )
    pairs = [(v,n) for n,v in doc_sims.items()]
    pairs.sort(reverse=True)
    options = [f"{round(v*100):3} {n.strip('.txt')}" for v,n in pairs]
    dpg.configure_item("SearchChoice",items=options,num_items=10)
    dpg.set_item_label("SearchButton","Search!")
    global LAST_DIR_SEARCHED
    LAST_DIR_SEARCHED = dpg.get_value("Directory")

def show_result():
    to_show = dpg.get_value("SearchChoice").strip()
    title = to_show[to_show.find(" "):].strip()
    filename = title + ".txt"
    directory = LAST_DIR_SEARCHED
    path = os.path.join(directory, filename)
    with open(path, "r") as file:
        text = file.read()
    dpg.set_value("DispTitle",title)
    print(f"Showing {title}")
    dpg.delete_item("DispTextGroup",children_only=True)
    sentences = FileManagement.split_text(text)
    for i, sentence in enumerate(sentences):
        dpg.add_text(sentence,parent="DispTextGroup")
    doc_embs = FileManagement.embed_or_load_dir(directory)[filename]
    sent_sims = BERT.term_similarity(
        pos_term=dpg.get_value("Positive Term"),
        neg_term=dpg.get_value("Negative Term"),
        embedded_documents={i: doc_embs[i] for i in range(len(doc_embs))},
        neg_weight=dpg.get_value("Negative Weight"),
        strategy=dpg.get_value("Method")
    )
    dpg.delete_item("DispTextGroup",children_only=True)
    black = (255,255,255)
    green = (124,252,0)
    for i, sentence in enumerate(sentences):
        color = [black[j]*(1 - sent_sims[i]**2) + green[j]*(sent_sims[i]**2) for j in range(3)]
        dpg.add_text(sentence+".",parent="DispTextGroup",color=color)

with dpg.window(
    tag="Search",
    no_collapse=True,
    no_close=True,
    no_title_bar=True,
    menubar=True,
    no_resize=True,
    no_move=True,
):
    dpg.add_listbox(["KBLab/sentence-bert-swedish-cased","deepset/sentence_bert"],label="BERT Model",num_items=2,tag="BERT Model")
    dpg.add_input_text(label="Directory",tag="Directory",default_value="SvenskaNoveller")
    dpg.add_input_text(label="Positive Term",tag="Positive Term")
    dpg.add_input_text(label="Negative Term",tag="Negative Term")
    dpg.add_slider_float(label="Negative Weight",tag="Negative Weight",max_value=1,default_value=0.5)
    dpg.add_listbox(["Max","Mean","Top3"],label="Method",num_items=3,tag="Method")
    dpg.add_button(label="Search!",tag="SearchButton",callback=search,width=150,height=75)

with dpg.window(
    tag="SearchResults",
    no_collapse=True,
    no_close=True,
    no_title_bar=True,
    menubar=True,
    no_resize=True,
    no_move=True,
):
    dpg.add_text("Search Results")
    dpg.add_listbox([],tag="SearchChoice",callback=show_result)

with dpg.window(
    tag="Text",
    no_collapse=True,
    no_close=True,
    no_title_bar=True,
    menubar=True,
    no_resize=True,
    no_move=True,
):
    dpg.add_text("",tag="DispTitle")
    dpg.add_text("",tag="DispText")
    dpg.add_group(tag="DispTextGroup")

def position_windows():
    # Split space between windows
    width = dpg.get_viewport_client_width()
    height = dpg.get_viewport_client_height()
    dpg.set_item_width("Search",int(0.25*width))
    dpg.set_item_width("SearchResults",int(0.25*width))
    dpg.set_item_width("Text",int(0.75*width))
    dpg.set_item_height("Search",int(height/2))
    dpg.set_item_height("SearchResults",int(0.5*height))
    dpg.set_item_height("Text",height)
    dpg.set_item_pos("Search",[0,0])
    dpg.set_item_pos("SearchResults",[0,int(0.5*height)])
    dpg.set_item_pos("Text",[int(0.25*width),0])
    # Place the "Search" button
    sb_width = dpg.get_item_width("SearchButton")
    sb_y_pos = dpg.get_item_pos("Method")[1] + 100
    dpg.set_item_pos("SearchButton",[int(sb_width/2),sb_y_pos])
    # Give the "SearchChoice" space
    dpg.set_item_width("SearchChoice",int(0.25*width))
dpg.set_viewport_resize_callback(position_windows)

dpg.create_viewport(title='Sem-Search', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
position_windows()
dpg.start_dearpygui()

dpg.destroy_context()