class ProtSeqExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProtSeqExplorer v0.0.1")
        self.resize(1000, 700)

        main_window = QWidget()

        self.prot_seq_label = QLabel("Protein Sequences")
        self.prot_plot_label = QLabel("Dimensionality Reduced Plot of Protein Sequence Embeddings")

        self.model = QStandardItemModel()  # to show the protein sequences and their names
        self.model.setHorizontalHeaderLabels(["Name", "Sequence", "Label"])
        self.prot_seq_tree = QTreeView()   # maybe use SQL and table later when also considering annotations
        self.prot_seq_tree.setModel(self.model)
        self.prot_seq_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.prot_seq_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.prot_seq_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.prot_seq_tree.setStyleSheet("QTreeView { border: 2px solid black; }")

        
        self.load_seqs_button = QPushButton("Load Sequences")
        self.process_seqs_button = QPushButton("Preprocess Seqeunces")  # preprocess, deal with missing or ambiguous residues
        self.save_button = QPushButton("Save Results")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.emb_method_label = QLabel("Embedding Methods:")
        self.emb_method_checkbox = MultiChoiceCheckBox(["EEV", 'ANV'])
        self.dim_red_method_label = QLabel("Dimensionality Reduction Methods:")
        self.dim_red_method_method_checkbox = MultiChoiceCheckBox(["PCA", 'UMAP', "DensMAP", 'TSNE'])

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
        
        self.load_seqs_button.clicked.connect(self.open_parse_file)
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
        self.anv = AANaturalVector()
        self.pca = PCA(n_components=2, svd_solver='covariance_eigh', n_jobs=2)
        self.umap = umap.UMAP(n_components=2, n_jobs=4, metric='euclidean')
        self.densmap = umap.UMAP(n_components=2, n_jobs=4, metric='euclidean', densmap=True)
        self.tsne = TSNE(n_components=2, n_jobs=2)


    def open_parse_file(self):
        self.sequences = []   # all info cleared when the button is pressed, may need to put this after the filedialog
        self.sequence_names = []
        self.sequence_labels = []
        
        file_path, _ = QFileDialog.getOpenFileName(filter="Accepted file formats (*.fasta *.csv *.tsv)")

        if file_path.endswith(".fasta"):
            self.model.clear()
            self.model.setHorizontalHeaderLabels(["Name", "Sequence", "Label"])
            self.parse_fasta(file_path)
        elif file_path.endswith("sv"):
            self.model.clear()
            self.model.setHorizontalHeaderLabels(["Name", "Sequence", "Label"])
            self.parse_csv(file_path)
        else:
            return

        self.check_sequence_validity()
        self.change_tree_border_color()

        self.process_seqs_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.emb_plot_button.setEnabled(True)
        self.clear_button.setEnabled(True)

    
    def parse_fasta(self, fasta_file_path):
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
            self.model.appendRow([seq_name, seq_show, QStandardItem("")])  # no label


    def parse_csv(self, file_path):
        if file_path.endswith(".csv"):
            delimiter=','
        else:
            delimiter = '\t'
        with open(file_path, newline='') as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            col_names = list(next(reader).keys())
            # choose the sequence column, name column, label column
            first_two_rows = []  # get the first two rows
            for j, row in enumerate(reader):
                if j < 2:
                    first_two_rows.append(list(row.values()))
                else:
                    break

            # get no. of unique values per column
            unqiue_values_per_column = [0] * len(col_names)
            for row in reader:
                
                    
            self.csv_tsv_col_choice(col_names, first_two_rows)

            if self.seq_col:  # a choice was made
                for i, row in enumerate(reader):
                    seq = row[self.seq_col].strip()
                    self.sequences.append(seq)
                    
                    if self.seq_name_col == 'Use index as name':
                        seq_name = i
                        self.sequence_names.append(seq_name)
                    else:
                        seq_name = row[self.seq_name_col]
                        self.sequence_names.append(row[self.seq_name_col])
                        
                    if self.seq_label_col == 'No labels':
                        seq_label = ''
                        self.sequence_labels.append(seq_label)
                    else:
                        seq_label = row[self.seq_label_col]
                        self.sequence_labels.append(row[self.seq_label_col])
                    
                    self.model.appendRow([QStandardItem(seq_name), 
                                          QStandardItem(f'{seq[:10]}...'), 
                                          QStandardItem(f'{seq_label[:15]}...')])  # show first 15
            else:
                return
                    

            
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

    def csv_tsv_col_choice(self, columns: list, first_two_rows):  # chooses sequence col, sequence name col, sequence label col from CSV/TSV
        dialog = QDialog(self)
        dialog.setWindowTitle("Parse CSV/TSV file")
        layout = QVBoxLayout(dialog)
        
        sneakpeak = QHBoxLayout()  # first two roles of the csv/tsv file with header
        model = QStandardItemModel()  # to show the protein sequences and their names
        sneakpeak_tree = QTreeView()
        sneakpeak_tree.setModel(model)
        model.setHorizontalHeaderLabels(columns)
        model.appendRow([QStandardItem(item) for item in first_two_rows[0]])
        model.appendRow([QStandardItem(item) for item in first_two_rows[1]])
        sneakpeak.addWidget(sneakpeak_tree)

        # number of unique values for each column
        
        hbox = QHBoxLayout()
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()
        col1.addWidget(QLabel('Sequence Column:'))
        col1.addWidget(QLabel('Sequence Name Column:'))
        col1.addWidget(QLabel('Sequence Label Column:'))
        self.seq_col_ComboBox = QComboBox()
        self.seq_col_ComboBox.addItems(columns)
        col2.addWidget(self.seq_col_ComboBox)
        self.seq_name_ComboBox = QComboBox()
        self.seq_name_ComboBox.addItems(['Use index as name'] + columns)
        col2.addWidget(self.seq_name_ComboBox)
        self.seq_label_ComboBox = QComboBox()
        self.seq_label_ComboBox.addItems(['No labels'] + columns)
        col2.addWidget(self.seq_label_ComboBox)

        hbox.addLayout(col1)
        hbox.addLayout(col2)
        layout.addLayout(sneakpeak)
        layout.addLayout(hbox)
        
        # Add OK / Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.seq_col = self.seq_col_ComboBox.currentText()
            self.seq_name_col = self.seq_name_ComboBox.currentText()
            self.seq_label_col = self.seq_label_ComboBox.currentText()

        else:
            self.seq_col, self.seq_name_col, self.seq_label_col = '', '', ''
        

    def embed_and_plot(self):
        selected_embs = self.emb_method_checkbox.get_selected_values()  # list
        selected_dim_reds = self.dim_red_method_method_checkbox.get_selected_values()  # list
        embs_mapping = {
            'EEV': lambda seq: self.eev.seq2vector(seq),
            'ANV': lambda seq: self.anv.seq2vector(seq),
                }
        dim_red_mapping = {
            'PCA': self.pca,#.fit_transform(array),
            'UMAP': self.umap,
            'DensMAP': self.densmap,
            'TSNE': self.tsne
                }
        if selected_embs and selected_dim_reds:
            self.figure.clear()
            self.canvas.draw()
            combinations = list(product(selected_embs, selected_dim_reds))
            n_combinations = len(combinations)

            cols = min(4, n_combinations)
            rows = math.ceil(n_combinations / cols)


            if n_combinations > 1:
                axes = self.figure.subplots(rows, cols)
                axes = axes.flatten()
                for i, (emb_name, dim_red_name) in enumerate(combinations):
                    embeddings = np.array(list(map(embs_mapping[emb_name], self.sequences)))   # everytime need calculate the embedding again, maybe need to think of a way to cache
                    reduced = dim_red_mapping[dim_red_name].fit_transform(embeddings)
                    
                    axes[i].scatter(reduced[:, 0], reduced[:, 1], color='blue', alpha=0.5, s=30)
                    axes[i].set_title(f'{emb_name} + {dim_red_name}')
                    axes[i].set_xlabel(f'{dim_red_name} 1')
                    axes[i].set_ylabel(f'{dim_red_name} 2')
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
    
                for j in range(i + 1, len(axes)):
                    self.figure.delaxes(axes[j])

            else:
                axes = self.figure.subplots()
                embeddings = np.array(list(map(embs_mapping[selected_embs[0]], self.sequences)))   # everytime need calculate the embedding again, maybe need to think of a way to cache
                reduced = dim_red_mapping[selected_dim_reds[0]].fit_transform(embeddings)
                
                axes.scatter(reduced[:, 0], reduced[:, 1], color='blue', alpha=0.5, s=30)
                axes.set_title(f'{selected_embs[0]} + {selected_dim_reds[0]}')
                axes.set_xlabel(f'{selected_dim_reds[0]} 1')
                axes.set_ylabel(f'{selected_dim_reds[0]} 2')
                axes.set_xticks([])
                axes.set_yticks([])

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
        layout = QHBoxLayout()

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
