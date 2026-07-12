import { useState } from 'react'
import './index.css'

function App() {
  const [url, setUrl] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [processStatus, setProcessStatus] = useState(null)
  const [question, setQuestion] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [chatHistory, setChatHistory] = useState([])

  const handleProcessVideo = async (e) => {
    e.preventDefault()
    if (!url) return
    setIsProcessing(true)
    setProcessStatus(null)
    setChatHistory([])
    try {
      const response = await fetch('https://agile-sparkle-production-d50b.up.railway.app/api/process-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      })
      const data = await response.json()
      if (response.ok) {
        setProcessStatus({ type: 'success', message: data.message })
      } else {
        setProcessStatus({ type: 'error', message: data.detail || 'Error processing video' })
      }
    } catch (err) {
      setProcessStatus({ type: 'error', message: 'Failed to connect to backend' })
    } finally {
      setIsProcessing(false)
    }
  }

  const handleAskQuestion = async (e) => {
    e.preventDefault()
    if (!question) return
    
    const newChat = [...chatHistory, { type: 'user', text: question }]
    setChatHistory(newChat)
    setQuestion('')
    setIsAsking(true)
    
    try {
      const response = await fetch('https://agile-sparkle-production-d50b.up.railway.app/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })
      const data = await response.json()
      
      if (response.ok) {
        setChatHistory([...newChat, { type: 'bot', text: data.answer }])
      } else {
        setChatHistory([...newChat, { type: 'bot', text: `Error: ${data.detail}` }])
      }
    } catch (err) {
      setChatHistory([...newChat, { type: 'bot', text: 'Error connecting to backend' }])
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>YouTube RAG Assistant</h1>
        <p>Ask questions directly to your favorite YouTube videos</p>
      </header>
      
      <main className="main-content">
        <section className="setup-section">
          <form onSubmit={handleProcessVideo} className="input-form">
            <input 
              type="text" 
              placeholder="Paste YouTube URL here..." 
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="url-input"
            />
            <button type="submit" disabled={isProcessing} className="primary-btn">
              {isProcessing ? 'Processing...' : 'Process Video'}
            </button>
          </form>
          
          {processStatus && (
            <div className={`status-message ${processStatus.type}`}>
              {processStatus.message}
            </div>
          )}
        </section>

        {processStatus?.type === 'success' && (
          <section className="chat-section">
            <div className="chat-window">
              {chatHistory.length === 0 ? (
                <div className="empty-chat">
                  <p>Video loaded successfully! Ask me anything about it.</p>
                </div>
              ) : (
                chatHistory.map((chat, idx) => (
                  <div key={idx} className={`chat-bubble ${chat.type}`}>
                    <div className="chat-content">{chat.text}</div>
                  </div>
                ))
              )}
              {isAsking && (
                <div className="chat-bubble bot typing">
                  <div className="chat-content">Thinking...</div>
                </div>
              )}
            </div>
            
            <form onSubmit={handleAskQuestion} className="chat-input-form">
              <input 
                type="text"
                placeholder="Ask a question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="chat-input"
              />
              <button type="submit" disabled={isAsking || !question} className="send-btn">
                Send
              </button>
            </form>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
