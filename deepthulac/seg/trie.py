class TrieNode:
    """ A node in the trie structure """

    def __init__(self):
        """ Node initialization """
        self.children = {}  # Dictionary to hold the children nodes
        self.is_end_of_word = False  # Flag to mark the end of a word

class Trie:
    """ The trie structure """

    def __init__(self):
        """ Trie initialization """
        self.root = TrieNode()

    def insert(self, word):
        """ Function to insert a word into the trie """
        node = self.root
        for char in word:
            # If character is not present in children of the current node, add it
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True  # Mark the end of the word

    def predict(self, prefix):
        """ Function to predict words starting with a given prefix """
        node = self.root
        for char in prefix:
            # If the character is not found, return an empty list
            if char not in node.children:
                # return []
                return False
            node = node.children[char]
        
        return True

        # Recursive function to find all words starting with the given node
        # def dfs(node, prefix, words):
        #     if node.is_end_of_word:
        #         words.append(prefix)
        #     for char, child in node.children.items():
        #         dfs(child, prefix + char, words)

        # predictions = []
        # dfs(node, prefix, predictions)
        # return predictions

    def __contains__(self, word):
        """ Check if a word exists in the trie """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word