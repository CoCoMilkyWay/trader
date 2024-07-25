#pragma once

#include <functional>

// ���������
class ScopeGuard
{
public:
  explicit ScopeGuard( std::function < void() > p_onExitScope )
    : m_onExitScope( p_onExitScope )
    , m_bDismissed( false )
  {
  }

  ~ScopeGuard()
  {
    if ( !m_bDismissed )
    {
      m_onExitScope();
    }
  }

  void Dismiss() // �������
  {
    m_bDismissed = true;
  }

private:
  std::function < void() > m_onExitScope;
  bool m_bDismissed;

private: // noncopyable
  ScopeGuard(ScopeGuard const&) = delete;
  ScopeGuard& operator=(ScopeGuard const&) = delete;
};

