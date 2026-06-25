from typing import cast

from sklearn.model_selection import train_test_split

from tool_forge.schema import Conversation

type Conversations = list[Conversation]

def split(conversations: Conversations, seed: int = 0) -> tuple[Conversations, ...]:
    """Split the conversations into ~80/10/10 train, val, test splis"""
    trainval, test = train_test_split(conversations, test_size=0.10, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.111, random_state=seed)
    
    train = cast(Conversations, train)
    val = cast(Conversations, val)
    test = cast(Conversations, test)

    return (train, val, test)

