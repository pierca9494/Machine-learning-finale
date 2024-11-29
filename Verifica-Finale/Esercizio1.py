class Libro:
    def __init__(self, titolo, autore, anno, quantita):
        self.titolo = titolo
        self.autore = autore
        self.anno = anno
        self.quantita = quantita

    def __str__(self):
        return f"{self.titolo} di {self.autore}, {self.anno} (Quantità: {self.quantita})"

class Libreria:
    def __init__(self):
        self.libri = []

    def aggiungi_libro(self, titolo, autore, anno, quantita):
        self.libri.append(Libro(titolo, autore, anno, quantita))

    def visualizza_libri(self):
        if not self.libri:
            print("Nessun libro in libreria.")
        else:
            for idx, libro in enumerate(self.libri, start=1):
                print(f"{idx}. {libro}")

    def cerca_libro_per_titolo(self, titolo):
        for libro in self.libri:
            if libro.titolo.lower() == titolo.lower():
                return libro
        return None

    def gestisci_libro(self, titolo):
        libro = self.cerca_libro_per_titolo(titolo)
        if libro:
            print(f"Libro trovato: {libro}")
            print("1. Rimuovi libro")
            print("2. Modifica libro")
            print("3. Aggiungi copie")
            scelta = input("Scegli un'opzione: ")
            if scelta == "1":
                self.libri.remove(libro)
                print("Libro rimosso.")
            elif scelta == "2":
                libro.titolo = input("Inserisci nuovo titolo: ") or libro.titolo
                libro.autore = input("Inserisci nuovo autore: ") or libro.autore
                libro.anno = input("Inserisci nuovo anno: ") or libro.anno
                libro.quantita = int(input("Inserisci nuova quantità: ") or libro.quantita)
                print("Libro aggiornato.")
            elif scelta == "3":
                try:
                    copie_da_aggiungere = int(input("Quante copie vuoi aggiungere? "))
                    if copie_da_aggiungere > 0:
                        libro.quantita += copie_da_aggiungere
                        print(f"Aggiunte {copie_da_aggiungere} copie.")
                    else:
                        print("Il numero di copie deve essere maggiore di zero.")
                except ValueError:
                    print("Inserisci un numero valido.")
            else:
                print("Opzione non valida.")
        else:
            print("Libro non trovato.")

def main():
    libreria = Libreria()

    while True:
        print("\nMenu Libreria:")
        print("1. Aggiungi un nuovo libro")
        print("2. Visualizza tutti i libri")
        print("3. Cerca un libro per titolo")
        print("4. Gestisci libri")
        print("5. Esci")

        scelta = input("Scegli un'opzione: ")

        if scelta == "1":
            titolo = input("Inserisci il titolo del libro: ")
            autore = input("Inserisci l'autore del libro: ")
            anno = input("Inserisci l'anno di pubblicazione: ")
            quantita = int(input("Inserisci la quantità: "))
            libreria.aggiungi_libro(titolo, autore, anno, quantita)
            print("Libro aggiunto.")

        elif scelta == "2":
            print("\nLibri in Libreria:")
            libreria.visualizza_libri()

        elif scelta == "3":
            titolo = input("Inserisci il titolo da cercare: ")
            libro = libreria.cerca_libro_per_titolo(titolo)
            if libro:
                print(f"Libro trovato: {libro}")
            else:
                print("Libro non trovato.")

        elif scelta == "4":
            titolo = input("Inserisci il titolo da gestire: ")
            libreria.gestisci_libro(titolo)

        elif scelta == "5":
            print("Uscita dal programma.")
            break

        else:
            print("Scelta non valida. Riprova.")

if __name__ == "__main__":
    main()
