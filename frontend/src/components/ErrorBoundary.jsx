import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    console.error(error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="fatal-error">
          <h1>Не удалось отрисовать приложение</h1>
          <p>Откройте консоль браузера, чтобы посмотреть детали runtime-ошибки.</p>
        </main>
      );
    }

    return this.props.children;
  }
}
