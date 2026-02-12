class ProtSeqExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProtSeqExplorer v0.0.1")
        self.resize(800, 600)

        main_window = QWidget()

        self.prot_seq_label = QLabel("Protein Sequences")
        self.prot_plot_label = QLabel("Dimensionality Reduced Plot of Protein Sequence Embeddings")

        self.model = QStandardItemModel()  # to show the protein sequences and their names
        self.model.setHorizontalHeaderLabels(["Name", "Sequence"])
        self.prot_seq_tree = QTreeView()   # maybe use SQL and table later when also considering annotations
        self.prot_seq_tree.setModel(self.model)
        self.prot_seq_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.prot_seq_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.prot_seq_tree.setStyleSheet("QTreeView { border: 2px solid black; }")

        
        self.load_seqs_button = QPushButton("Load Sequences")
        self.process_seqs_button = QPushButton("Preprocess Seqeunces")  # preprocess, deal with missing or ambiguous residues
        self.save_button = QPushButton("Save Results")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.emb_method_label = QLabel("Embedding Methods:")
        self.emb_method_checkbox = MultiChoiceCheckBox(["EEV"])
        self.dim_red_method_label = QLabel("Dimensionality Reduction Methods:")
        self.dim_red_method_method_checkbox = MultiChoiceCheckBox(["PCA"])

        #self.emb_button = QPushButton("Embed!")
        #self.plot_buttonn = QPushButton("Plot!")
        self.emb_plot_button = QPushButton("Embed and Plot!")
        self.clear_button = QPushButton("Clear")


        self.master_layout = QVBoxLayout()
        self.row1 = QHBoxLayout()
        self.row2 = QHBoxLayout()
        self.row3 = QHBoxLayout()
        self.row4 = QHBoxLayout()
        self.tree_col = QVBoxLayout()
        self.canvas_col = QVBoxLayout()
        self.button_col1 = QVBoxLayout()
        self.button_col2 = QVBoxLayout()
        self.methods_row = QHBoxLayout()
        self.method_labels_col = QVBoxLayout()
        self.method_checkboxes_col = QVBoxLayout()

        self.row1.addWidget(self.prot_seq_label)
        self.row1.addWidget(self.prot_plot_label)
        
        self.tree_col.addWidget(self.prot_seq_tree)
        self.canvas_col.addWidget(self.canvas)
        self.row2.addLayout(self.tree_col, 20)
        self.row2.addLayout(self.canvas_col, 80)

        
        self.button_col1.addWidget(self.load_seqs_button)
        self.button_col1.addWidget(self.process_seqs_button)
        self.method_labels_col.addWidget(self.emb_method_label)
        self.method_labels_col.addWidget(self.dim_red_method_label)
        self.method_checkboxes_col.addWidget(self.emb_method_checkbox)
        self.method_checkboxes_col.addWidget(self.dim_red_method_method_checkbox)
        self.row3.addLayout(self.button_col1, 20)
        self.row3.addLayout(self.method_labels_col, 20)
        self.row3.addLayout(self.method_checkboxes_col, 60)

        self.row4.addWidget(self.save_button)
        self.row4.addWidget(self.emb_plot_button)
        self.row4.addWidget(self.clear_button)
        

        self.master_layout.addLayout(self.row1)
        self.master_layout.addLayout(self.row2)
        self.master_layout.addLayout(self.row3)
        self.master_layout.addLayout(self.row4)

        main_window.setLayout(self.master_layout)
        self.setCentralWidget(main_window)
        
        self.load_seqs_button.clicked.connect(self.open_parse_fasta)
        self.process_seqs_button.clicked.connect(self.process_sequences_window)
        self.emb_plot_button.clicked.connect(self.embed_and_plot)
        self.clear_button.clicked.connect(self.clear)
        
        self.process_seqs_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.emb_plot_button.setEnabled(False)
        self.clear_button.setEnabled(False)

        self.sequences = []
        self.sequence_names = []
        self.sequences_valid = None
        self.eev = EnergyEntropy_1(data_type='protein')
        self.pca = PCA(n_components=2)


    def open_parse_fasta(self):
        self.sequences = []
        self.sequence_names = []
        
        fasta_file_path, _ = QFileDialog.getOpenFileName(filter="Fasta files (*.fasta)")
        with open(fasta_file_path, 'r') as f:
            fasta_content = f.read()
            
        pattern = r'^>([^\n]+)\n(.*?)(?=^>|\Z)'
        matches = re.findall(pattern, fasta_content, re.MULTILINE | re.DOTALL)
        
        headers = []
        sequences = []
        
        for header, seq in matches:
            self.sequence_names.append(header)
            # Remove newlines from sequence to get a single continuous string
            self.sequences.append(seq.replace('\n', ''))

            seq_name = QStandardItem(str(header))
            seq_show = QStandardItem(f'{seq.replace('\n', '')[:10]}...')  # only show the first 10 residues
            self.model.appendRow([seq_name, seq_show])

        self.check_sequence_validity()
        self.change_tree_border_color()

        self.process_seqs_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.emb_plot_button.setEnabled(True)
        self.clear_button.setEnabled(True)

    def check_sequence_validity(self):
        bad_residues = set('XBZUO')
        for seq in self.sequences:
            if bad_residues.intersection(set(seq)):
                self.sequences_valid = False
                break
        self.sequences_valid = True

    def change_tree_border_color(self):
        if self.sequences_valid:
            self.prot_seq_tree.setStyleSheet("QTreeView { border: 2px solid green; }")
        else:
            self.prot_seq_tree.setStyleSheet("QTreeView { border: 2px solid red; }")

    def process_sequences_window(self):
        # no functionality yet
        msgBox = QMessageBox(self)
        msgBox.setIcon(QMessageBox.Icon.Question)
        msgBox.setText("What do you want to do with them?")
        msgBox.setWindowTitle("Preprocess Sequences")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msgBox.setDefaultButton(QMessageBox.StandardButton.Ok)

        # 2. Create the QCheckBox instance
        self.checkbox = QCheckBox(":p", msgBox)
        # 3. Add the checkbox to the QMessageBox using setCheckBox()
        msgBox.setCheckBox(self.checkbox)
        # 4. Execute the dialog and get the user's button choice
        ret = msgBox.exec()
        # 5. Check the state of the checkbox after the dialog is closed
        if ret == QMessageBox.StandardButton.Ok:
            if self.checkbox.isChecked():
                print("OK clicked and checkbox was checked.")
                # You can save this preference (e.g., to a config file)
            else:
                print("OK clicked and checkbox was not checked.")
        elif ret == QMessageBox.StandardButton.Cancel:
            print("Cancel clicked.")

    def embed_and_plot(self):
        selected_embs = self.emb_method_checkbox.get_selected_values()  # list
        selected_dim_reds = self.dim_red_method_method_checkbox.get_selected_values()  # list
        embs_mapping = {
                    'EEV': lambda seq: self.eev.seq2vector(seq),
                }
        dim_red_mapping = {
                    'PCA': self.pca#.fit_transform(array),
                }
        if selected_embs and selected_dim_reds:
            self.figure.clear()
            self.canvas.draw()
            
            if len(selected_embs) == 1 and len(selected_dim_reds) == 1:
                embeddings = np.array(list(map(embs_mapping[selected_embs[0]], self.sequences)))
                reduced = dim_red_mapping[selected_dim_reds[0]].fit_transform(embeddings)
                
                self.figure.clear()
                ax = self.figure.subplots()
    
                ax.scatter(reduced[:, 0], reduced[:, 1], color='blue', alpha=0.5, s=30)
                ax.set_title(f'{selected_dim_reds[0]}')
                ax.set_xlabel(f'{selected_dim_reds[0]} 1')
                ax.set_ylabel(f'{selected_dim_reds[0]} 2')
                ax.set_xticks([])
                ax.set_yticks([])
                self.canvas.draw()

        else:
            QMessageBox.warning(self, "Embed and Plot!", "No embedding or dimensionality reduction method selected.")

    def clear(self):
        self.figure.clear()
        self.canvas.draw()
                


class MultiChoiceCheckBox(QWidget):
    def __init__(self, options: list):
        super().__init__()
        assert type(options) == list
        layout = QVBoxLayout()

        self.checkboxes = []

        for option in options:
            checkbox = QCheckBox(option, self)
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        self.setLayout(layout)

    def get_selected_values(self) -> list:
        selected_values = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                # Get the text associated with the checkbox
                selected_values.append(checkbox.text())
        
        return selected_values

if __name__=='__main__':
    app = QApplication([])
    my_app = ProtSeqExplorer()
    my_app.show()
    app.exec()
    gc.collect()
