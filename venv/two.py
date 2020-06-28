from random import sample
ranks = ['2', '3', '4', '5','6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
suits = ['Spades', 'Clubs','Diamonds','Hearts']

"""
Input : integers rank , suit
 Output : tuple comprised of input values
"""
def createCard(rank,suit):
    return(rank,suit)
'''
13 Create an ordered list of cards, i.e. [(0,0),(0,1),...(1,0)
,...]
'''
def createDeck():
    deck=[]
    for el in range(len(ranks)):
        for su in range(len(suits)):
            deck.append(createCard(el,su))
    return deck

'''
Randomly shuffle deck 
'''
def shuffleDeck(deck):
    return(sample(deck,len(deck)))
'''
Input: n, positive integer
Returns a list of n first cards from deck, which are removed from deck
'''
def dealFromDeck(deck, n):
    lst = []
    if len(deck) >=n:
        for i in range(n):
            lst.append(deck.pop())
        return(lst)
    else:
        print("not enough cards in deck")
        return (None)

'''
Input: a list of cards
Output: prints out what those cards are 
'''
def printHand(p):
    for i in p:
        print(ranks[i[0]]+ " "+'of'+' '+ suits[i[1]])

n_players = 5 # Number of players
oDeck = createDeck ()
deck = shuffleDeck(oDeck)

# Deal 3 cards to each player
player_hands = []
for p in range(n_players):
    hand = dealFromDeck(deck ,3)
    player_hands.append(hand)

print(createDeck())

# Print the hands
for p in player_hands:
    printHand(p)