class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatboxRoot: document.querySelector('.chatbox'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
        this.state = false;
        this.messages = [];
    } 
    
    display() {
        const {openButton, chatboxRoot, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatboxRoot));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })    
    }

    toggleState(chatboxRoot) {
        this.state = !this.state;
        chatboxRoot.classList.toggle('chatbox--active', this.state);
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value.trim();
        if (text1 === "") {
            return;
        }   
        let msg1 = { name: "User", message: text1 } 
        this.messages.push(msg1);
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Bot", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = '';
        })
        .catch((error) => {
            console.error('Error:', error);
            let msg2 = { name: "Bot", message: "Sorry, I could not reach the chatbot right now." };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = '';
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item) {
            const div = document.createElement('div');
            div.className = item.name === "Bot"
                ? 'messages__item messages__item--visitor'
                : 'messages__item messages__item--operator';
            div.textContent = item.message;
            if (item.name === "Bot") {
                html += div.outerHTML
            }
            else {
                html += div.outerHTML
            }
        });
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();
