import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css"; // Make sure you have CSS for animations

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("en");
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("idle"); // "idle", "listening", "processing", "busy"
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const chatBoxRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  const startRecording = async () => {
    if (status === "busy") return; // Prevent starting recording if busy

    try {
      setStatus("listening");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        setStatus("processing");
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        await sendAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      setStatus("idle"); // Reset status to idle on error
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudio = async (audioBlob) => {
    setStatus("busy"); // Set status to busy to prevent multiple requests

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    formData.append("language", language === "he" ? "Hebrew" : "en");

    try {
      const response = await axios.post("http://localhost:5000/transcribe", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setInput(response.data.text); // Live update of recognized words
      setStatus("idle"); // Reset circle to waiting
    } catch (error) {
      console.error("Error transcribing audio:", error);
      setStatus("idle"); // Reset status to idle on error
      audioChunksRef.current = []; // Clear audio chunks on error
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
  
    const newMessages = [...messages, { text: input, sender: "user" }];
    setMessages(newMessages);
    setInput("");
  
    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });
  
      if (!response.ok) {
        throw new Error("Failed to fetch bot response.");
      }
  
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let botMessage = { text: "", sender: "bot" };
      setMessages([...newMessages, botMessage]);
  
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        botMessage.text += decoder.decode(value, { stream: true });
        setMessages([...newMessages, botMessage]);
      }
    } catch (error) {
      console.error("Error while sending message:", error);
      alert("An error occurred while trying to get a response from the bot.");
    }
  };

  const handleLanguageSwitch = (newLanguage) => {
    setLanguage(newLanguage);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };
  
  return (
    <div className="chat-container">
      <div className="chat-box">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender === "user" ? "user-message" : "bot-message"}`}>
            <div className="message-text">{msg.text}</div>
          </div>
        ))}
      </div>

      <div className="language-switcher">
        <div
          className={`switch-container ${language === "he" ? "he-active" : "en-active"}`}
          onClick={() => handleLanguageSwitch(language === "he" ? "en" : "he")}
        >
          <div className="switch-button"></div>
        </div>
        <div className="language-label">
          {language === "he" ? "he" : "en"}
        </div>
      </div>

      <div className="input-container">
        {/* Circle button */}
        <div
          className={`circle ${status}`}
          onMouseDown={status === "idle" ? startRecording : null}
          onMouseUp={status === "listening" ? stopRecording : null}
        ></div>

        <textarea
          ref={inputRef}
          className="input-box"
          value={input}
          onChange={handleInputChange}
          placeholder="Type a message..."
          onKeyDown={handleKeyDown}
          rows={1}
        />
      </div>
    </div>
  );
}

export default App;
